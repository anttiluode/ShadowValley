"""Shadow Valley in a tiny transformer.

HEAD (anttiluode/HEAD) found a trap in a 323-parameter reader: when the target is a
deviation from a context average and another path already carries that average,
"look away from the answer and flip the decoder sign" is also a valid solution.
Gradient descent from a negative initial decoder sign slides into that valley and
stalls partway. Neon's formula for a head spread evenly over m wrong slots:

    best decoder b = -m/(8-m),     NMSE = 1 - m/(7(8-m)).

The general form for any weight vector a over the 8 slots (signed or not) is

    NMSE(a) = 1 - cos^2(a - mean(a), e_j - 1/8)          (zero-sum detail)

which reduces to Neon's curve for uniform-over-m, zero-on-j. This file asks
whether ordinary transformer blocks fall into the same valley, and whether
three remedies escape it: Differential attention, Cog attention, and unsigned
"flashlight" routing (HEAD's early auxiliary, softmax unchanged).

Task (HEAD's, in token form). 8 slot tokens x_i = c + d_i, plus one query token.
c is one of 8 cue levels; a hidden per-seed permutation maps cue -> slot j.
Target y = polarity * d_j.
  detail 'zero-sum': d = g - mean(g), g ~ N(0,1)^8  -> mean(x) = c exactly
  detail 'iid'     : d ~ N(0, 7/8) i.i.d.           -> no look-away solution exists

Architectures
  A  'attn'  : 1 attention layer, no LayerNorm, no MLP, linear readout. The query
               token embeds the cue (one-hot) and c. HEAD's reader written as a
               transformer; the cos^2 formula is exact for its head output.
  B  'block' : 1 standard pre-LN block (attention + GELU MLP), query token embeds cue.
  A' 'attn-scalar', B' 'block-scalar': the same, but the query sees only the scalar cue
               level c (HEAD's condition: a low-capacity query), no one-hot cue embedding.
  C  'self'  : 2 standard pre-LN blocks, query token gets NO cue. The network must
               compute the average itself (layer 1) and route from it (layer 2).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

SLOTS = 8
LEVELS = torch.tensor([2 * k / (SLOTS - 1) - 1 for k in range(SLOTS)])


# ----------------------------------------------------------------- task
def sample(n, mapping, gen, detail='zero-sum', polarity=1.0):
    k = torch.randint(SLOTS, (n,), generator=gen)
    c = LEVELS[k]
    g = torch.randn(n, SLOTS, generator=gen)
    d = g - g.mean(1, keepdim=True) if detail == 'zero-sum' else g * math.sqrt(7 / 8)
    x = c[:, None] + d
    j = mapping[k]
    y = polarity * d.gather(1, j[:, None]).squeeze(1)
    return x, y, k


# ----------------------------------------------------------------- attention
class Attention(nn.Module):
    """Multi-head attention with the four routing rules. Returns output and the
    query-token row of weights/logits for every head (for measurement)."""

    def __init__(self, d, heads, kind):
        super().__init__()
        self.h, self.dh, self.kind = heads, d // heads, kind
        nq = 2 if kind == 'diff' else 1
        self.q = nn.Linear(d, d * nq, bias=False)
        self.k = nn.Linear(d, d * nq, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if kind == 'diff':
            # Ye et al. 2024 re-parameterisation, lambda_init 0.8 (GroupNorm omitted, see README)
            self.lam_init = 0.8
            self.lq1, self.lk1, self.lq2, self.lk2 = (nn.Parameter(torch.randn(self.dh) * 0.1) for _ in range(4))

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.h, self.dh
        v = self.v(x).view(B, T, H, dh).transpose(1, 2)
        if self.kind == 'diff':
            q = self.q(x).view(B, T, 2, H, dh).permute(2, 0, 3, 1, 4)
            k = self.k(x).view(B, T, 2, H, dh).permute(2, 0, 3, 1, 4)
            s1 = q[0] @ k[0].transpose(-1, -2) / math.sqrt(dh)
            s2 = q[1] @ k[1].transpose(-1, -2) / math.sqrt(dh)
            lam = torch.exp(self.lq1 @ self.lk1) - torch.exp(self.lq2 @ self.lk2) + self.lam_init
            a = s1.softmax(-1) - lam * s2.softmax(-1)
            logits = s1
            out = (a @ v) * (1 - self.lam_init)
        else:
            q = self.q(x).view(B, T, H, dh).transpose(1, 2)
            k = self.k(x).view(B, T, H, dh).transpose(1, 2)
            s = q @ k.transpose(-1, -2) / math.sqrt(dh)
            if self.kind == 'cog':          # Lv et al. 2024: sign(s) * softmax(|s|)
                a = torch.sign(s) * s.abs().softmax(-1)
            else:                            # 'softmax' and 'flashlight'
                a = s.softmax(-1)
            logits = s
            out = a @ v
        out = self.o(out.transpose(1, 2).reshape(B, T, D))
        # query token is position 0; slots are 1..8
        return out, a[:, :, 0, 1:], logits[:, :, 0, 1:]


class Block(nn.Module):
    def __init__(self, d, heads, kind, norm=True, mlp=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d) if norm else nn.Identity()
        self.att = Attention(d, heads, kind)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) if mlp else None

    def forward(self, h):
        o, a, s = self.att(self.ln1(h))
        h = h + o
        if self.mlp is not None:
            h = h + self.mlp(h)
        return h, a, s


class TinyTransformer(nn.Module):
    def __init__(self, arch, kind, heads=1, d=32):
        super().__init__()
        self.arch = arch
        self.slot_x = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(SLOTS, d) * 0.5)
        self.qtok = nn.Parameter(torch.randn(d) * 0.5)
        # '-scalar' archs: the query sees only the cue level c (HEAD's condition), no one-hot
        self.cue = nn.Embedding(SLOTS, d) if arch in ('attn', 'block') else None
        self.cue_c = nn.Linear(1, d, bias=False) if arch != 'self' else None
        base = arch.replace('-scalar', '')
        if base == 'attn':
            self.blocks = nn.ModuleList([Block(d, heads, kind, norm=False, mlp=False)])
            self.ln_f = nn.Identity()
        elif base == 'block':
            self.blocks = nn.ModuleList([Block(d, heads, kind)])
            self.ln_f = nn.LayerNorm(d)
        else:
            self.blocks = nn.ModuleList([Block(d, heads, kind), Block(d, heads, kind)])
            self.ln_f = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, x, cue_ids):
        B = x.shape[0]
        slots = self.slot_x(x[..., None]) + self.pos
        q = self.qtok.expand(B, -1)
        if self.cue is not None:
            q = q + self.cue(cue_ids)
        if self.cue_c is not None:
            q = q + self.cue_c(LEVELS[cue_ids][:, None])
        h = torch.cat([q[:, None], slots], 1)
        for blk in self.blocks:
            h, a, s = blk(h)
        # a, s: last layer's query-token row, (B, heads, 8)
        return self.out(self.ln_f(h[:, 0])).squeeze(1), a, s


# ----------------------------------------------------------------- flashlight teacher
class CorrTable:
    """HEAD's unsigned protocol: running Pearson correlation, per cue and slot, between
    the centred slot reading x_i - mean(x) and the detached residual y - y_hat."""

    def __init__(self):
        self.s = torch.zeros(6, SLOTS, SLOTS, dtype=torch.float64)

    def collect(self, cue, z, r):
        for i in range(SLOTS):
            zi = z[:, i].double(); rr = r.double()
            for t, val in enumerate((torch.ones_like(zi), zi, rr, zi * zi, rr * rr, zi * rr)):
                self.s[t, :, i].index_add_(0, cue, val)

    def targets(self, mode):
        n, sx, se, sxx, see, sxe = self.s
        cov = sxe / n - sx * se / n ** 2
        corr = cov / torch.sqrt((sxx / n - (sx / n) ** 2) * (see / n - (se / n) ** 2) + 1e-30)
        score = corr ** 2 if mode == 'unsigned' else corr
        return score.argmax(1)


# ----------------------------------------------------------------- one run
def nmse(p, y):
    return float(((p - y) ** 2).mean() / (y ** 2).mean())


def predicted_nmse(a_row, j, detail):
    """Closed form for a linear readout of this head given the cue: a_row (8,), slot j."""
    t = -torch.ones(SLOTS) / SLOTS; t[j] += 1
    if detail == 'zero-sum':
        u = a_row - a_row.mean()
        return float(1 - (u @ t) ** 2 / ((u @ u) * (t @ t) + 1e-30))
    return float(1 - a_row[j] ** 2 / (a_row @ a_row + 1e-30))


def train_eval(seed, arch, kind, heads=1, detail='zero-sum', polarity=1.0, steps=2000, batch=256,
               lr=3e-3, aux_weight=0.25, aux_frac=0.25, aux_mode='unsigned', record=False):
    torch.manual_seed(seed)
    mapping = torch.randperm(SLOTS, generator=torch.Generator().manual_seed(1000 + seed))
    flash = kind == 'flashlight'
    model = TinyTransformer(arch, 'softmax' if flash else kind, heads)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.Generator().manual_seed(7000 + seed)
    shuf = torch.Generator().manual_seed(37000 + seed)
    table, table_t, prev = CorrTable(), None, None
    aux_steps = int(aux_frac * steps)
    trace = []
    for step in range(steps):
        x, y, k = sample(batch, mapping, data, detail, polarity)
        p, a, s = model(x, k)
        loss = F.mse_loss(p, y)
        if flash and step < aux_steps:
            r = (y - p).detach()
            z = x - x.mean(1, keepdim=True)
            if aux_mode == 'shuffled':
                # Pair this batch's readings with residuals of the SAME cue from the PREVIOUS batch.
                # (A within-batch permutation leaks: its ~1 fixed point per cue keeps ~1/n of the
                # true correlation, enough for the teacher to find the slot.)
                cur = (k.clone(), z.clone(), r.clone())
                if prev is None:
                    prev = cur
                else:
                    pk, pz, pr = prev
                    ks, zs, rs = [], [], []
                    for g in range(SLOTS):
                        a_ix, b_ix = torch.where(k == g)[0], torch.where(pk == g)[0]
                        n = min(len(a_ix), len(b_ix))
                        ks.append(k[a_ix[:n]]); zs.append(z[a_ix[:n]]); rs.append(pr[b_ix[:n]])
                    prev = cur
                    table.collect(torch.cat(ks), torch.cat(zs), torch.cat(rs))
            else:
                table.collect(k, z, r)
            if step >= 8:
                if aux_mode == 'random':     # control: a fixed random cue->slot table, no information
                    tgt = torch.randperm(SLOTS, generator=torch.Generator().manual_seed(55000 + seed))
                else:
                    tgt = table.targets('signed' if aux_mode == 'signed' else 'unsigned')
                loss = loss + aux_weight * F.cross_entropy(s[:, 0], tgt[k])
                table_t = tgt
        opt.zero_grad(); loss.backward(); opt.step()
        if record and (step in (0, 10, 50, 100, 300) or step % 500 == 0 or step == steps - 1):
            trace.append([step, float(loss), float(a[:, 0][torch.arange(batch), mapping[k]].mean())])

    model.eval()
    with torch.no_grad():
        x, y, k = sample(8192, mapping, torch.Generator().manual_seed(17000 + seed), detail, polarity)
        p, a, s = model(x, k)
        res = {'seed': seed, 'arch': arch, 'kind': kind, 'heads': heads, 'detail': detail,
               'polarity': polarity, 'nmse': nmse(p, y), 'mapping': mapping.tolist()}
        per_cue = []
        for g in range(SLOTS):
            ix = k == g; j = int(mapping[g])
            A = a[ix].mean(0)                  # (heads, 8)
            rows = []
            for hh in range(A.shape[0]):
                row = A[hh]
                others = torch.cat([row[:j], row[j + 1:]])
                po = others.abs() / others.abs().sum()
                u = row - row.mean(); tt = -torch.ones(SLOTS) / SLOTS; tt[j] += 1
                rows.append({'a_target': float(row[j]), 'top_is_target': int(row.argmax()) == j,
                             'abs_top_is_target': int(row.abs().argmax()) == j,
                             'cos_to_target': float(u @ tt / (u.norm() * tt.norm() + 1e-30)),
                             'bottom_is_target': int(row.argmin()) == j,
                             'm_eff_others': float(1 / (po ** 2).sum()),
                             'pred_nmse': predicted_nmse(row, j, detail)})
            per_cue.append(rows)
        res['per_cue'] = per_cue
        best = [max(range(heads), key=lambda hh: -per_cue[g][hh]['pred_nmse']) for g in range(SLOTS)]
        res['routes_correct'] = sum(any(per_cue[g][hh]['top_is_target'] for hh in range(heads)) for g in range(SLOTS))
        res['looks_away'] = sum(all(per_cue[g][hh]['bottom_is_target'] for hh in range(heads)) for g in range(SLOTS))
        res['a_target_mean'] = sum(per_cue[g][best[g]]['a_target'] for g in range(SLOTS)) / SLOTS
        res['m_eff_mean'] = sum(per_cue[g][best[g]]['m_eff_others'] for g in range(SLOTS)) / SLOTS
        res['pred_nmse_head0'] = sum(per_cue[g][0]['pred_nmse'] for g in range(SLOTS)) / SLOTS
        m = res['m_eff_mean']
        res['neon_curve_nmse'] = 1 - m / (7 * (8 - m)) if m < 8 else None
        # effective decoder sign: regress the prediction on head-0 read sum_i a_i x_i, per cue
        v = (a[:, 0] * x[:, ...]).sum(1)
        bs = []
        for g in range(SLOTS):
            ix = k == g
            vv, pp = v[ix] - v[ix].mean(), p[ix] - p[ix].mean()
            bs.append(float((vv * pp).sum() / ((vv * vv).sum() + 1e-30)))
        res['beta_eff_median'] = float(torch.tensor(bs).median())
        res['cos_mean'] = sum(per_cue[g][best[g]]['cos_to_target'] for g in range(SLOTS)) / SLOTS
        # how the head reads the target, per cue (best head): at / negative-at / away / other
        modes = []
        for g in range(SLOTS):
            q = per_cue[g][best[g]]
            modes.append('at' if q['abs_top_is_target'] and q['a_target'] > 0 else
                         'negative-at' if q['abs_top_is_target'] else
                         'away' if q['cos_to_target'] < 0 else 'other')
        res['route_modes'] = modes
        res['route_mode'] = max(set(modes), key=modes.count)
        res['success'] = res['nmse'] < 0.01
        # outcome: solved / valley (inverted side: the read points away) / other failure
        res['outcome'] = 'solved' if res['success'] else ('valley' if res['cos_mean'] < 0 else 'direct-side failure')
        if flash and table_t is not None:
            res['teacher_correct_at_cutoff'] = int((table_t == mapping).sum())
        if record:
            res['trace'] = trace
    return res
