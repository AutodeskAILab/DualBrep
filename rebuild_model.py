"""Rebuilder model — ``Parametrizer``.

Given per-face sampled points (from a segmented mesh), re-fits each B-rep face as a
parametric surface grid and predicts the face-face intersection edges and topology.
"""
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

def hermite_sample(v_face, v_uv):
    uv = v_uv[:, None] * 2 - 1
    uv = uv[..., [1, 0]]
    out = torch.nn.functional.grid_sample(
        v_face, uv,
        # mode="bicubic",
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )

    return out[:, :, 0]


def normalize_coord1112(v_points):
    points = v_points[..., :3]
    normals = v_points[..., 3:6]
    shape = points.shape
    num_items = shape[0]
    points = points.reshape(num_items, -1, 3)
    target_points = points + normals.reshape(num_items, -1, 3)

    center = points.mean(dim=1, keepdim=True)
    scale = (torch.linalg.norm(points - center, dim=-1)).max(dim=1, keepdims=True)[0]
    assert scale.min() > 1e-4
    points = (points - center) / (scale[:, None] + 1e-6)
    target_points = (target_points - center) / (scale[:, None] + 1e-6)
    normals = target_points - points
    normals = normals / (1e-6 + torch.linalg.norm(normals, dim=-1, keepdim=True))

    points = points.reshape(shape)
    normals = normals.reshape(shape)

    return points, normals, center[:, 0], scale


def denormalize_coord1112(points, bbox):
    normal = points[..., 3:]
    points = points[..., :3]
    target_points = points + normal
    center = bbox[..., :3]
    scale = bbox[..., 3:4]
    while len(points.shape) > len(center.shape):
        center = center.unsqueeze(1)
        scale = scale.unsqueeze(1)
    points = points * scale + center
    target_points = target_points * scale + center
    normal = target_points - points
    normal = normal / (1e-6 + torch.linalg.norm(normal, dim=-1, keepdim=True))
    points = torch.cat((points, normal), dim=-1)
    return points


def normalize_coord0516(v_points, eps=1e-2):
    points = v_points[..., :3]
    with_normals = (v_points.shape[-1] == 6)
    shape = points.shape
    num_items = shape[0]
    points = points.reshape(num_items, -1, 3)
    if with_normals:
        normals = v_points[..., 3:6]
        target_points = points + normals.reshape(num_items, -1, 3)

    center = points.mean(dim=1, keepdim=True)
    bbox = points.max(dim=1)[0] - points.min(dim=1)[0]

    bbox_pad = bbox.clone()
    bbox_pad[bbox_pad<eps] = 1
    points = (points - center) / bbox_pad[:, None]
    if with_normals:
        target_points = (target_points - center) / bbox_pad[:, None]
        normals = target_points - points
        normals = normals / (1e-6 + torch.linalg.norm(normals, dim=-1, keepdim=True))
        normals = normals.reshape(shape)

    points = points.reshape(shape)

    if with_normals:
        points = torch.cat((points, normals), dim=-1)
    bbox = torch.cat((center[:,0], bbox), dim=-1)
    return points, bbox


def denormalize_coord0516(v_points, bbox, eps=1e-2):
    with_normals = (v_points.shape[-1] == 6)
    points = v_points[..., :3]
    if with_normals:
        normal = v_points[..., 3:]
        target_points = points + normal
    center = bbox[..., :3]
    scale = bbox[..., 3:6].clone()
    scale[scale < eps] = 1
    while len(points.shape) > len(center.shape):
        center = center.unsqueeze(1)
        scale = scale.unsqueeze(1)
    points = points * scale + center
    if with_normals:
        target_points = target_points * scale + center
        normal = target_points - points
        normal = normal / (1e-6 + torch.linalg.norm(normal, dim=-1, keepdim=True))
        points = torch.cat((points, normal), dim=-1)
    return points


class MLP(nn.Module):
    def __init__(self, *,
                 width: int):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class MLP2(nn.Module):
    def __init__(self, inc, outc=None, dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = inc * 4
        self.c_fc = nn.Linear(inc, dim_feedforward)
        self.c_proj = nn.Linear(dim_feedforward, outc if outc is not None else inc)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.c_proj(self.relu(self.c_fc(x)))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
            self,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_3 = nn.LayerNorm(width)
        self.embedder = FourierEmbedder(num_freqs=8, include_input=True, input_dim=3, include_pi=False)
        self.input_proj = nn.Linear(self.embedder.out_dim + 3, width)
        self.output_proj = nn.Linear(width, 32)

    def forward(self, v_points):
        point_feat = self.embedder(v_points[..., :3])
        point_feat = torch.cat((point_feat, v_points[..., 3:6]), dim=-1)
        point_feat = self.input_proj(point_feat)

        seed_index = (torch.rand(v_points.shape[:2], device=v_points.device) * v_points.shape[1]).to(torch.int64)[:, :16]
        seed_feat = torch.gather(point_feat, 1, seed_index[:, :, None].repeat(1, 1, point_feat.shape[-1]))

        x = self.ln_2(seed_feat)
        x = x + self.attn(self.ln_1(x), point_feat, point_feat, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))
        x = self.output_proj(x)
        # x = x.reshape(x.shape[0], -1)
        return x
    

class ResidualQueryAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(width,heads,batch_first=True,)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        data = self.ln_2(data)
        x = x + self.attn(self.ln_1(x), data, data, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))
        return x


class ResidualQueryAttentionFPSBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width,heads,batch_first=True,)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        data = self.ln_2(data)
        x = x + self.attn(self.ln_1(x), data, data, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))
        return x



# Without layer norm
class PointEncoder3(nn.Module):
    def __init__(self, width, heads, num_seeds=2*2, num_in=6, num_out=32):
        super().__init__()
        self.num_seeds = num_seeds
        self.input_proj = nn.Linear(num_in, width)
        self.mlp = MLP2(width, width, 2048)

        self.query = ResidualQueryAttentionBlock(width, heads)
        self.fc = nn.Linear(width, num_out)

    def forward(self, v_points):
        point_feat = self.input_proj(v_points)
        idx = torch.rand_like(point_feat[..., 0]) * point_feat.shape[1]
        idx = idx.to(torch.long)[..., :self.num_seeds]
        seed_feat = torch.gather(point_feat, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, point_feat.shape[-1]))
        point_feat = self.query(seed_feat, point_feat)
        point_feat = point_feat + self.mlp(point_feat)
        point_feat = self.fc(point_feat)
        return point_feat


class residual_unconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, dim=2):
        super().__init__()
        if dim == 2:
            conv = nn.Conv2d
            unconv = nn.ConvTranspose2d
            self.inter_mode = 'bilinear'
        elif dim == 1:
            conv = nn.Conv1d
            unconv = nn.ConvTranspose1d
            self.inter_mode = 'linear'
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.unconv = unconv(out_channels, out_channels, kernel_size=2, stride=2) if upsample else nn.Identity()
        if in_channels != out_channels:
            self.res_conv = conv(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None
        self.interpolate = F.interpolate if upsample else None
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.unconv(out)
        if self.res_conv is not None:
            identity = self.res_conv(identity)
        identity = self.interpolate(identity, scale_factor=2, mode=self.inter_mode) if self.interpolate is not None else identity
        out += identity
        out = self.act(out)
        return out


class Parametrizer(nn.Module):
    def __init__(self, v_conf):
        super().__init__()
        self.loss_fn = nn.L1Loss() if v_conf["loss"] == "l1" else nn.MSELoss()
        aug_matrix = Rotation.create_group('O').as_matrix()
        self.register_buffer("aug_matrix", torch.from_numpy(aug_matrix).float())
        self.times = {
            "Augment": 0,
            "Encoder": 0,
            "Fuser": 0,
            "Sample": 0,
            "global": 0,
            "Decoder": 0,
            "Intersection": 0,
            "Loss": 0,
            "Encoder1": 0,
            "Encoder2": 0,
            "Encoder3": 0,
            "Encoder4": 0,
            "Decoder1": 0,
            "Decoder2": 0,
            "Decoder3": 0,
            "Decoder4": 0,
        }
        self.freeze_face = v_conf["freeze_face"]
        self.num_sample_points = v_conf["num_sample_points"]
        self.with_normal = v_conf["with_normal"]
        self.max_intersections = v_conf["max_intersections"]
        self.dim_shape = v_conf["dim_shape"]
        norm = "layer"
        ds = self.dim_shape

        self.in_channels = v_conf["in_channels"]
        dim = 256
        self.point_model = PointEncoder3(768, 8, num_in=6 if self.with_normal else 3, num_out=dim, num_seeds=2*2)

        bd = 768  # bottleneck_dim
        self.face_bbox_in = MLP2(6, dim, dim_feedforward=2048)
        layer = nn.TransformerEncoderLayer((2*2+1)*dim, 8, dim_feedforward=2048, dropout=0.1, batch_first=True, norm_first=True)
        self.face_attn = nn.TransformerEncoder(layer, 8, nn.LayerNorm((2*2+1)*dim))
        self.face_attn_proj_in = None
        self.face_attn_proj_out = None

        # Encoder
        self.face_norm_decoder_proj = None
        self.face_norm_decoder = nn.Sequential(
            residual_unconv_block(dim, 256, True),
            residual_unconv_block(256, 256, True),
            residual_unconv_block(256, 256, True),
            nn.Conv2d(256, 3 if not self.with_normal else 6, kernel_size=1, padding=0, stride=1)
        )

        # Face decoder
        self.face_center_scale_decoder = MLP2(dim*5, 6, dim_feedforward=2048)

        self.face_decode_proj_in = None
        self.uv_input_proj = None
        self.uv_querier = None
        self.uv_output_proj = None

        if self.max_intersections > 0:  
            self.inter = nn.Sequential(
                nn.Linear(((2*2+1)*dim) * 2, 2048),
                nn.LayerNorm(2048),
                nn.ReLU(),
                nn.Linear(2048, (2*256)),
            )
            self.classifier = nn.Sequential(
                nn.Linear((2*256), 2048),
                nn.LayerNorm(2048),
                nn.ReLU(),
                nn.Linear(2048, 1),
            )
            self.edge_norm_decoder = nn.Sequential(
                residual_unconv_block(256, 256, True, dim=1),
                residual_unconv_block(256, 256, True, dim=1),
                residual_unconv_block(256, 256, True, dim=1),
                nn.Conv1d(256, 5, kernel_size=1, padding=0, stride=1)
            )

            self.u_input_proj = None
            self.edge_decode_proj_in = None
            self.u_querier = None
            self.u_output_proj = None

        if v_conf["face_weights"] is not None:
            file = v_conf["face_weights"]
            with tempfile.TemporaryDirectory() as tmp:
                if file.startswith('s3'):
                    file = download_file(file, str(tmp))
                weight = torch.load(file, map_location='cpu', weights_only=False)["state_dict"]
            weight = {k[6:]: v for k, v in weight.items()}
            self.load_state_dict(weight, strict=False)
            print(f"Loaded face weights from {v_conf['face_weights']}.")

    def profile_time(self, timer, key):
        # torch.cuda.synchronize()
        if key not in self.times:
            self.times[key] = 0
        torch.cuda.current_stream().synchronize()
        self.times[key] += time.time() - timer
        timer = time.time()
        return timer

    def encode(self, v_data, v_test):
        timer = time.time()
        # Noisy the data
        # face_input_norm = v_data["face_input_norm"]
        face_input_bbox = v_data["face_input_bbox"]
        face_sample_points = v_data["face_sample_points"]

        if False:
            # use gt bbox to Normalize
            points = face_sample_points[..., :3]
            shape = points.shape
            num_items = shape[0]
            points = points.reshape(num_items, -1, 3)
            normals = face_sample_points[..., 3:6]
            target_points = points + normals.reshape(num_items, -1, 3)

            center = bbox[:, :3]
            bbox_pad = bbox[:, 3:].clone()
            bbox_pad[bbox_pad < 1e-2] = 1
            points = (points - center[:, None]) / bbox_pad[:, None]
            target_points = (target_points - center[:, None]) / bbox_pad[:, None]
            normals = target_points - points
            normals = normals / (1e-6 + torch.linalg.norm(normals, dim=-1, keepdim=True))
            normals = normals.reshape(shape)

            points = points.reshape(shape)
            points = torch.cat((points, normals), dim=-1)
        # face_norm, face_bbox = normalize_coord0516(face_sample_points)

        point_features = self.point_model(face_sample_points)  # [batch_size, 2*2, 256]
        bbox_feat = self.face_bbox_in(face_input_bbox)  # [batch_size, 256]
        point_features = torch.cat((point_features, bbox_feat[:, None]), dim=1)  # [batch_size, (2*2)+1, 256]
        point_features = point_features.reshape(point_features.shape[0], -1)  # [batch_size, (2*2+1) * 256]
        timer = self.profile_time(timer, "Encoder1")

        attn_x = self.face_attn(point_features, mask=v_data["face_attn_mask"])  # [batch_size, (2*2+1) * 256]

        attn_x = attn_x.reshape(attn_x.shape[0], 2*2+1, -1)  # [batch_size, (2*2)+1, 256]
        timer = self.profile_time(timer, "Encoder2")

        return {"face_z": attn_x, "face_bbox": face_input_bbox}

    def decode_face(self, encoding_result, uv_sample):
        face_z = encoding_result["face_z"]
        face_bbox = encoding_result["face_bbox"]
        decoding_results = {}

        timer = time.time()
        conv_feat = face_z[:, :2*2, :].reshape(face_z.shape[0], 2, 2, face_z.shape[-1]).permute(0, 3, 1, 2)
        decoding_results["face_norm"] = self.face_norm_decoder(conv_feat).permute(0, 2, 3, 1)
        timer = self.profile_time(timer, "Decoder1")

        delta_bbox = self.face_center_scale_decoder(face_z.reshape(face_z.shape[0],-1))
        decoding_results["face_bbox"] = delta_bbox + face_bbox
        # decoding_results["face_bbox"] = delta_bbox
        timer = self.profile_time(timer, "Decoder2")
        return decoding_results

    def query_edge(self, intersection_feat, u_sample):
        intersection_feat = intersection_feat.reshape(intersection_feat.shape[0], -1, 2)
        result = self.edge_norm_decoder(intersection_feat)
        return result.permute(0, 2, 1)

    def decode_edge(self, face_z, v_data=None):
        decoding_results = {}
        if v_data is None:
            num_faces = face_z.shape[0]
            device = face_z.device
            indexes = torch.stack(
                torch.meshgrid(
                    torch.arange(num_faces), torch.arange(num_faces), indexing="ij"
                ),
                dim=2,
            )

            indexes = indexes.reshape(-1, 2).to(device)
            num_edges = indexes.shape[0]
            feature_pair = face_z[indexes].reshape(num_edges, -1)
            if self.freeze_face:
                feature_pair = feature_pair.detach()
            intersected_edge_feature = []
            pred = []
            batch_size = 10000
            for item in feature_pair.split(batch_size, dim=0):
                result = self.inter(item)
                intersected_edge_feature.append(result)

                result = self.classifier(result)[..., 0]
                pred.append(result)

            intersected_edge_feature = torch.cat(intersected_edge_feature, dim=0)
            pred = torch.cat(pred, dim=0)
            pred_labels = torch.sigmoid(pred) > 0.5

            intersected_edge_feature = intersected_edge_feature[pred_labels]
            decoding_results["pred_face_adj"] = pred_labels.reshape(-1)
            decoding_results["pred_edge_face_connectivity"] = torch.cat(
                (torch.arange(intersected_edge_feature.shape[0], device=device)[
                 :, None], indexes[pred_labels],),
                dim=1,
            )
            decoding_results["face_index"] = indexes[pred_labels]
            decoding_results["edge_index"] = torch.arange(intersected_edge_feature.shape[0], device=device)
            u_sample = None
        else:
            edge_face_connectivity = v_data["edge_face_connectivity"]
            v_zero_positions = v_data["zero_positions"]

            if edge_face_connectivity.shape[0] > self.max_intersections * len(v_data["v_prefix"]):
                index = torch.randperm(edge_face_connectivity.shape[0])[:self.max_intersections * len(v_data["v_prefix"])]
                edge_face_connectivity = edge_face_connectivity[index]
                v_zero_positions = v_zero_positions[index]
            decoding_results["face_index"] = edge_face_connectivity[:, 1:]
            decoding_results["edge_index"] = edge_face_connectivity[:, 0]
            true_intersection_embedding = face_z[edge_face_connectivity[:, 1:]]
            false_intersection_embedding = face_z[v_zero_positions]
            id_false_start = true_intersection_embedding.shape[0]
            feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
            num_edges = feature_pair.shape[0]
            feature_pair = feature_pair.reshape(num_edges, -1)
            if self.freeze_face:
                feature_pair = feature_pair.detach()
            intersected_edge_feature = self.inter(feature_pair)
            pred = self.classifier(intersected_edge_feature)

            gt_labels = torch.ones_like(pred)
            gt_labels[id_false_start:] = 0
            loss_edge = F.binary_cross_entropy_with_logits(pred, gt_labels)

            intersected_edge_feature = intersected_edge_feature[:id_false_start]
            decoding_results["intersected_edge_feature"] = intersected_edge_feature
            decoding_results["loss_edge"] = loss_edge

            u_sample = None
        if intersected_edge_feature.shape[0] == 0:
            decoding_results["edge_uv"] = torch.zeros((0, 5), device=self.aug_matrix.device)
        else:
            decoding_results["edge_uv"] = self.query_edge(intersected_edge_feature, u_sample)
        return decoding_results

    def loss(self, v_decoding_result, v_data):
        timer = time.time()
        shape = v_decoding_result["face_norm"].shape
        loss = {}

        gt_norm = v_data["face_norm"][..., :3].contiguous()
        diff = torch.gradient(gt_norm, dim=[1, 2])
        # device = v_data["face_norm"].device
        # dtype = v_data["face_norm"].dtype
        # kernel_x = torch.zeros(3, 1, 1, 3, device=device, dtype=dtype)
        # kernel_x[:, 0, 0, 0] = -0.5
        # kernel_x[:, 0, 0, 2] = 0.5
        # kernel_y = torch.zeros(3, 1, 3, 1, device=device, dtype=dtype)
        # kernel_y[:, :, 0, 0] = -0.5
        # kernel_y[:, :, 2, 0] = 0.5
        # gt_norm = gt_norm.permute(0, 3, 1, 2)
        # diff1 = F.conv2d(gt_norm, kernel_x, padding=(0, 1), groups=3).permute(0, 2, 3, 1)
        # diff2 = F.conv2d(gt_norm, kernel_y, padding=(1, 0), groups=3).permute(0, 2, 3, 1)
        # diff = [diff2, diff1]
        timer = self.profile_time(timer, "Loss11")
        dis1 = torch.linalg.norm(diff[0], dim=-1)
        dis2 = torch.linalg.norm(diff[1], dim=-1)
        timer = self.profile_time(timer, "Loss12")
        dis = torch.min(dis1, dis2)
        dis = torch.clamp(dis, 0.001, 1)
        # dis[:, [0,-1]]=1
        # dis[:, :, [0,-1]]=1
        # Assign higher loss with lower distance
        face_loss = F.l1_loss(v_decoding_result["face_norm"], v_data["face_norm"], reduction="none")
        face_adpt_loss = (face_loss/dis[..., None]).mean()
        timer = self.profile_time(timer, "Loss13")

        estimated_diff = torch.gradient(v_decoding_result["face_norm"], dim=[1, 2])
        estimated_normal = torch.linalg.cross(estimated_diff[0][..., :3], estimated_diff[1][..., :3])
        estimated_normal = F.normalize(estimated_normal, dim=-1)
        predicted_normal = v_decoding_result["face_norm"][..., 3:]
        delta = 1-(predicted_normal*estimated_normal).sum(dim=-1).mean()
        timer = self.profile_time(timer, "Loss2")

        loss["face_normal"] = delta
        loss["face_ori_coord"] = face_loss.mean()
        loss["face_adpt_coord"] = face_adpt_loss
        loss["face_norm"] = face_adpt_loss+loss["face_normal"]*0.01
        loss["face_bbox"] = self.loss_fn(v_decoding_result["face_bbox"], v_data["face_bbox"])
        timer = self.profile_time(timer, "Loss3")

        if self.max_intersections > 0:
            loss["edge_classification"] = v_decoding_result["loss_edge"] * 0.1
            fi = v_decoding_result["face_index"]
            ei = v_decoding_result["edge_index"]
            
            pred_edge_uv = v_decoding_result["edge_uv"][:, :, :2]
            loss["edge_uv"] = self.loss_fn(pred_edge_uv, v_data["edge_uv"][ei][:, :, :2],)

            pred_edge_norm = v_decoding_result["edge_uv"][:, :, 2:5]
            gt_bbox = v_data["face_bbox"][fi[:, 0]]
            pred_edge_points = denormalize_coord0516(pred_edge_norm, gt_bbox)[..., :3]
            loss["edge_points"] = self.loss_fn(pred_edge_points, v_data["edge_points"][ei],)
            timer = self.profile_time(timer, "Loss4")

        return loss

    def forward(self, v_data, v_test=False):
        torch.cuda.current_stream().synchronize()
        timer = time.time()
        # v_data = augment3(v_data, v_test, self.aug_matrix, v_num_sample_points=self.num_sample_points, v_with_normal=self.with_normal)
        if not v_test:
            face_sample_points = v_data["face_sample_points"]
            face_sample_points[..., :3] = face_sample_points[..., :3] + torch.randn_like(face_sample_points[..., :3]) * 0.001
            face_sample_points[..., 3:] = F.normalize(face_sample_points[..., 3:] + torch.randn_like(face_sample_points[..., 3:]) * 0.05, dim=-1)
            v_data["face_input_bbox"] += torch.randn_like(v_data["face_input_bbox"]) * 0.001
        # v_data["face_input_norm"], v_data["face_input_bbox"] = normalize_coord0516(v_data["face_sample_points"])

        timer = self.profile_time(timer, "Augment")

        encoding_result = self.encode(v_data, v_test)
        timer = self.profile_time(timer, "Encoder")
        decoding_result = self.decode_face(encoding_result, None)
        timer = self.profile_time(timer, "Decoder")
        if self.max_intersections > 0:
            decoding_result2 = self.decode_edge(encoding_result["face_z"], v_data)
            decoding_result.update(decoding_result2)
            timer = self.profile_time(timer, "Decoder edge")
        loss = self.loss(decoding_result, v_data)
        # for idx, item in enumerate(decoding_result["face_norm"]):
        #     import trimesh
        #     trimesh.PointCloud(item[..., :3].detach().cpu().numpy().reshape(-1,3)).export(f"face_{idx}.ply")
        timer = self.profile_time(timer, "Loss")
        loss["total_loss"] = sum(loss.values())
        data = {}
        if v_test:
            gt_face_norm = v_data["face_norm"]
            # pred_face = self.inference_grid_face(encoding_result)
            pred_face = denormalize_coord0516(decoding_result["face_norm"], decoding_result["face_bbox"])
            gt_face = denormalize_coord0516(gt_face_norm, v_data["face_bbox"])
            loss["face_coords"] = nn.functional.l1_loss(pred_face[..., :3], gt_face[..., :3])

            data["gt_face"] = gt_face.detach().cpu().numpy()
            data["pred_face"] = pred_face.detach().cpu().numpy()

            if self.max_intersections > 0:
                # Compute edge
                fi = decoding_result["face_index"]
                ei = decoding_result["edge_index"]
                gt_edge = v_data["edge_points"][ei]

                pred_edge_uv = decoding_result["edge_uv"][..., :2]
                pred_edge = hermite_sample(pred_face[fi[:, 0]][..., :3].permute(0, 3, 1, 2),pred_edge_uv)
                pred_edge = pred_edge.permute(0, 2, 1)
                loss["edge_coords"] = nn.functional.l1_loss(pred_edge[..., :3],gt_edge[..., :3],)

                pred_edge_norm = decoding_result["edge_uv"][..., 2:5]
                pred_bbox = decoding_result["face_bbox"][fi[:, 0]]
                pred_edge_points = denormalize_coord0516(pred_edge_norm, pred_bbox)[..., :3]
                loss["edge_coords2"] = nn.functional.l1_loss(pred_edge_points, gt_edge[..., :3])
                data["gt_edge"] = gt_edge.detach().cpu().numpy()
                data["pred_edge1"] = pred_edge.detach().cpu().numpy()

                # Compute F1
                pred_data = self.decode_edge(encoding_result["face_z"])
                fi = pred_data["face_index"]
                num_gt_faces = data["gt_face"].shape[0]
                face_adj = torch.zeros((num_gt_faces,num_gt_faces), dtype=bool, device=loss["total_loss"].device)
                conn = v_data["edge_face_connectivity"]
                face_adj[conn[:, 1], conn[:, 2]] = True
                if fi.shape[0] == 0:
                    data["pred_edge"] = np.zeros((0, 16, 3))
                else:
                    pred_edge_uv = pred_data["edge_uv"][..., :2]
                    pred_edge = hermite_sample(pred_face[fi[:, 0]][..., :3].permute(0, 3, 1, 2), pred_edge_uv)
                    pred_edge = pred_edge.permute(0, 2, 1)
                    data["pred_edge"] = pred_edge.detach().cpu().numpy()
                data["gt_face_adj"] = face_adj.reshape(-1)
                data["pred_face_adj"] = pred_data["pred_face_adj"].reshape(-1)
                data["gt_edge_face_connectivity"] = (v_data["edge_face_connectivity"].detach().cpu().numpy())
                data["pred_edge_face_connectivity"] = (pred_data["pred_edge_face_connectivity"].detach().cpu().numpy())
        
        return loss, data

    def inference(self, v_data):
        encoding_result = self.encode(v_data, True)
        decoding_result = self.decode_face(encoding_result, None)
        pred_face = denormalize_coord0516(decoding_result["face_norm"], decoding_result["face_bbox"])
        result = {"pred_face": pred_face.detach().float().cpu().numpy()}
        if self.max_intersections > 0:
            decoding_result2 = self.decode_edge(encoding_result["face_z"])
            fe = decoding_result2["face_index"]
            pred_edge = torch.zeros((0, 3), device=self.aug_matrix.device, dtype=torch.float32)
            if decoding_result2["edge_uv"].shape[0] > 0:
                pred_edge = hermite_sample(pred_face[fe[:, 0]][..., :3].permute(0, 3, 1, 2),decoding_result2["edge_uv"][...,:2])
                pred_edge = pred_edge.permute(0, 2, 1)
            result["pred_edge"] = pred_edge.detach().float().cpu().numpy()
            result["pred_edge_face_connectivity"] = decoding_result2["pred_edge_face_connectivity"].detach().float().cpu().numpy()

        return result

    def decode_edge_given_topo(self, v_face_idx, v_face_feat):
        feature_pair = v_face_feat[v_face_idx]
        intersected_edge_feature = self.inter(feature_pair.reshape(feature_pair.shape[0],-1))
        edge_uv = self.query_edge(intersected_edge_feature, None)
        return edge_uv


