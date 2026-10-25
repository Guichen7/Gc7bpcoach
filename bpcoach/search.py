# -*- coding: utf-8 -*-
"""M4 阵容推荐：树搜索下一手（默认等深全展开：我方取最大、对方按 M2 加权），
M2 当先验与剪枝，M3 当叶子估值。

论文 M4 的骨架
    论文用 UCT 做树搜索，奖励定义为**剩余各局中赢下的局数**（不是单局胜率之和），
    因为 BoN 的胜利条件是赢 (N+1)/2 局。本模块保留这个骨架，但有三处必须讲清楚的改动。

一、为什么奖励要换成「系列赛晋级概率」
    先给一个数学事实：把单局胜率 p 做任何**单调变换**（比如乘剩余局数、或折成晋级概率），
    在**同一局之内**都不会改变候选的排序。所以「奖励 = 剩余局胜场」和「奖励 = 单局胜率」
    在同一个局面下选出的是同一手棋，差别只在数字怎么读。

    真正会随比分变化的是「这一局的分量」（swing = 赢下它的晋级概率 − 输掉它的晋级概率）。
    Bo5、p=0.5 时它长这样（自算，见 tests/test_search.py）：

        0:0 → 37.5%     1:0 → 37.5%     1:1 → 50%
        2:0 → 25%       2:1 → 50%       2:2 → 100%

    注意它**不是单调的**：领先 2:0 的那一局是全系列赛最不值钱的一局——赢了收工，
    输了大不了 2:1（还领先且握有两个赛点）；最贵的是决胜局，其次才是开局。
    这条直接决定了「什么时候该把强势英雄交出去」，也是下面「英雄池代价」的乘数。

二、为什么还要一个「英雄池代价」项
    上面说了，单调变换不改排序。那么「第几局、什么比分」怎么可能影响推荐？
    只有一条路：让奖励里出现**与具体英雄绑定**的项。全局禁选赛制下，这一局用掉的英雄
    后面几局就没了，所以「花掉一张强势卡」是有代价的；代价的大小取决于后面还有多少
    「更值钱的局」——用 swing 的补数来度量：

        奖励 = 晋级概率 − w · (1 − swing) · Σ price(这一局我方用掉的英雄)

    swing 越小（这一局越不值钱），扣得越狠，也就越倾向把强势英雄留到后面；
    决胜局 swing=1，代价归零，该梭哈就梭哈。
    反直觉但符合算术的一例：0:2 的第三局 swing 只有 25%，比 1:2 的第四局（50%）还小，
    所以「0:2 时留一手给第四局」在这套口径下是成立的——前提是你相信那 50%。
    这是启发式（一个旋钮，不是从数据估出来的参数），**默认关闭**（w=0）。

三、对手强度基线怎么进来
    M3 的 team 版模型里 `team_strength_diff` 的系数是 +3.18（logit 尺度）。
    把它乘上两队强度差，就得到一条与阵容无关的 logit 平移：p_eff = σ(logit(p) + βΔ)。
    注意它**只平移数值水平、不改变同局排序**（仍是 p 的单调变换）——这点必须写清楚，
    否则很容易被误读成「模型会因为对手强弱改主意」。它真正的作用有两个：
      ① 报告出来的晋级概率贴近这场比赛的现实（打重庆狼队和打弱队的 50% 不是一回事）；
      ② 与「英雄池代价」相乘时，swing 的形状会随 p 变，间接影响「该不该现在花英雄」。

四、搜索细节（以及为什么默认用等深全展开而不是 UCT）
    节点值统一记成「我方视角的奖励」，行动方是我方时取最大、是对方时按 M2 概率加权
    （默认口径，为什么不是「取最小」见六）。候选只取 M2 的前 k 个 —— M2 剪枝。
    叶子估值：从该节点用确定性补完走到 20 手，再用 M3 打分。
    不用 M3 给半成品阵容打分，是因为「3 个英雄的胜率」与「5 个英雄的胜率」不同量纲，
    混在同一棵树的回溯里会让浅层节点系统性偏乐观。

    UCT 也实现了（纯 numpy，不依赖 scipy），但**默认算法是等深全展开（期望极小极大）**。
    这里要更正一条早期的判断：曾经观察到「访问 407 次的分支值 0.62、访问 31 次的 0.36，
    差 26 个点」，并把它归因于 UCT 的均值回传。把补完策略修好之后（见 _rollout），
    这个现象**不复现**：同一根节点上 mcts(480 次) 与精确解的候选排序 ρ 中位数 0.95、
    值的平均绝对误差 0.41 个点、top-1 一致 4/6（`tools/eval_m4.py` 第 ③ 栏）。原因也不难说清：
      · 那 26 个点来自**值域被撑开**。补完策略糟糕时，候选值能从 0.26 排到 0.55，
        UCT 于是把预算全压在高值分支上；而被压得深的分支又因为我方多算了几层
        「最优选人」（对手始终只能在他们的 M2 前 4 手里挑）而更乐观，自我强化。
      · 补完策略修好后值域收窄到 3 个点，探索项（c_explore=0.55）压过 exploitation，
        访问自然趋于均匀，偏差随之消失。
    换句话说：那 26 个点是补完策略的病，不是 UCT 的病。
    默认仍用等深全展开，理由换成更朴素的一条——这棵树很小
    （我方 6 宽 × 对手 4 宽、深度 3 层 = 240 个叶子，全展开约 0.5 秒），
    精确值不花额外代价，也就没理由接受「统计上差不多」。
    `algorithm="mcts"` 保留：需要更深战术线（max_depth=24）或要按预算截断时用它。

五、禁人步为什么必须换尺子（拒绝价值）
    「给我一个禁人建议」是这个项目里教练最需要的功能，而上面那套价值函数给不出来。
    实测（`tools/audit_m4.py --check ban`）：我方禁人步 30 个样本，6/30 所有候选完全同值，
    值域中位数不到 1 个点；禁掉对手最强候选后，确定性补完的最终阵容只有 12/30 会变。
    数学根因有两层：
      · 我方禁人不改变我方那五个人，而价值函数只给我方阵容打分；
      · `d_prior_win` 是「我方均值 − 对方均值」的差分，禁掉最强者**双方都拿不到**，
        差分几乎不动——补完策略对双方是同一套，去掉一张卡后两边一起顺移。

    所以禁人步的排序依据换成**拒绝价值**。定义（方向最容易写反，写死在这里，
    p 一律是**我方**胜率）：
        denial(h) = p(对手失去 h、换上最好的替补) − p(对手拿到其最强阵容)  ≥ 0
    读法：「禁掉这张卡，我的胜率能回升多少」＝「把他们的天花板压掉多少」。
    「对手最强阵容」＝让 M3 判定的我方 p 最低的那五个人，用「强度 + 分路缺口」
    铺一个种子、再逐槽做局部搜索求出来（约 80ms）。这里**必须**做爬坡：
    第一版只铺了一次贪心，结果 M3 眼里那套种子并不强（它还把 2 个摇摆位塞了进去，
    d_flex 是负系数、d_lanes_ll 要求阵容能干净地排成 1-1-1-1-1），于是出现
    「禁掉大乔、对手反而变强」的荒唐值（威胁 p 从 0.549 → 0.522，算出来的 denial 是负的）。
    爬坡到局部最优后，阵容里不存在「拿掉会变强」的卡，denial 才非负，符号也才对。

    实测这个尺子与「禁最强」不是同义改写：拒绝价值最高的那张卡经常不是我方先验最强卡，
    也不是 M2 的首选（M2 给的 6 个候选，拒绝价值全部是 0——那些是我方大概率要抢的卡，
    禁掉等于自损）。所以禁人步的候选集也换了：不再用 M2 的前 k 手，直接对全池的
    「对手上限阵容」逐位评估。

    量级要说清楚：实测单张 ban 的拒绝价值上限约 0.8 个点（赵云这类），多数卡是 0.0x 个点，
    130 张里有 110~130 张禁了等于空过。这不是实现不够好，是这套模型的诚实结论——
    英雄池深、替补质量接近时，一次禁人对天花板的压缩本来就只有零点几个点，
    而 M3 的噪声底是 ±3 个点。它能回答的是「禁谁最不亏」，不能回答「禁谁能赢」。

    局限：① 对手上限用局部搜索求，不保证全局最优，所以 denial 是真实值的一个**下界**；
    ② 替补预筛取前 32 张（按强度 + 分路似然），漏掉真替补会略微高估 denial；
    ③ 只给「他们可能拿到的卡」加权，不给「他们大概会怎么打」加权，会高估冷门但适配的卡。

六、对手节点：最坏情况，还是按 M2 加权
    这个旋钮只管**对方行动**的节点怎么汇总（我方节点永远是取最大）。最初写的是
    min，理由听起来很稳：「对手总能挑到对我方最不利的那一手」。问题在于它和这棵树
    的其他假设**不是同一个世界**：
      · 候选集是 M2 的前 4 手 —— 用 M2 说「他会在这些里面挑」；
      · 叶子补完是 M2 的 top-1 —— 用 M2 说「他大概会这么打」；
      · 到汇总时突然改成「他会挑最狠的」，等于同一个预测器在三个地方被赋予三种人格。
    实测两种口径下首选有 5/29 不同（`tools/audit_m4.py --check opponent`），
    首选的价值差中位 0.82 个点、最大 2.76 个点。方向很清楚：min 让推荐偏「稳」——
    给「对手恰好按最坏方式应」的那一手加分；期望让推荐偏「强」——按对手实际会怎么打
    来算账。M2 的 top-1 只有 17.8%，也就是说对手大多数时候并不打你替他算的那一手，
    用 min 等于为一个小概率分支付全价。

    有一点必须澄清：换成期望**不会**让深度更稳定（d3 与 d4 选同一手：
    期望 24/30，最坏 25/30，见 `--check depth`）。所以这不是数值修复，是**口径选择**：
        opp_risk = 0    按 M2 概率加权（默认）
        opp_risk = 1    纯最坏情况（旧行为，等于原教旨极小极大）
        0 < x < 1       线性混合：v = (1−x)·E + x·min
    什么时候该调大：你认为真实对手比 M2 更聪明（M2 学的是「平均会怎么打」，
    没学「被打疼了会怎么调整」），或者这是输不起的一场，稳健比期望值重要。
    只想问「这套阵容客观上多强」，就用 opp_risk=0。
    副作用：MCTS 的选择规则也跟着走（见 _select）——期望口径下子节点值本身已经是
    「他各种应法的加权平均」，再翻一次符号就成了双重对抗。

七、这个模块**不能**做什么
    M3 本身 LOSO 只有 54.7%（AUC 0.54），所以 M4 的推荐是「在这个模型眼里的最优」，
    不是「客观上最优」。它的价值在于：把「补哪条路、留谁给后面、对手会怎么应」
    这三件事的权衡显式化、可复现、可辩论。评估里给的是「相对基线强多少」，
    不是「能提高多少胜率」。

    其中最大的一条能力缺口是**英雄交互（克制关系）**，这里记成明确决策，不是待办：

    M3 的叶子估值只有 6 个特征（d_prior_win / d_prior_mvp / d_lanes_ll / d_flex /
    d_score_diff / d_game_index），全都是「我方一套 vs 对方一套」的加减，
    **没有「这张卡克制那张卡」这一维**。所以不管树搜索多深、候选多宽，
    它都买不到克制——`tools/audit_m4.py --check reward` 量过：
    候选差异 52.5% 来自 d_prior_win、32.6% 来自 MVP、10.3% 来自分路，
    搜索实际在做的是「排分路 + 挑强度」，不是「找对位」。

    注意别和 M2 的跨队项混淆：M2 的 logLift(x, 对方已选) 确实是克制项，
    但它服务的是「猜对手下一手会选谁」，不参与「这套阵容多强」。同一个词，两处含义。

    为什么不做（三个方案都算过账）：
    1. one-hot 交互：C(134,2)=8911 个自由参数，而 M3 的训练样本是 **1447**（差分形式，
       一局一个样本）→ 平均 0.16 个样本/参数。连秩都估不出来。
    2. 低秩近似（同路对位，rank=4 → 134×4=536 参数）：约 2.7 样本/参数，
       数值上能跑，但 M3 的 LOSO 是 54.7%、95% CI 52.2%~57.2%（下限贴着基线 50%），
       而交互项能带来的量级远小于 ±3 个点的噪声——负结果可预期，也证不了伪。
    3. 引外部对位胜率数据：这是唯一真能补的路径，但它把结论的依赖从
       「我们自己 1447 局」换成「第三方的口径与版本」，需要先解决数据版本对齐，
       成本远超收益。

    结论：**先不做**。要让 M4 有克制感，正确顺序是先让 M3 具备识别克制的分辨率
    （本局版本感知的强度先验、或更大的样本），而不是在 54.7% 的估值上加一层
    它看不见的交互——那样只是把噪声重新排列成一张更精致的表。
"""

from __future__ import annotations

import math
import pathlib
import random
import time
from dataclasses import dataclass, field

from .errors import BPCoachError
from .heroes import HeroPool
from .series import BLUE, RED, other
from .state import DraftState
from .template import BAN, PICK, BPTemplate

EPS = 1e-9


# ====================== 系列赛数学 ======================
def clinch_probability(p: float, games_left: int, need: int) -> float:
    """还剩 games_left 局、还差 need 胜时，晋级（赢下系列赛）的概率。

    假设每局胜率都是 p 且相互独立——这是「这套阵容/这个水平能维持」的口径，
    不是对对手适应、体力、心态的建模。
    """
    if need <= 0:
        return 1.0
    if games_left <= 0 or need > games_left:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    return sum(
        math.comb(games_left, k) * (p ** k) * ((1 - p) ** (games_left - k))
        for k in range(need, games_left + 1)
    )


def expected_wins(p: float, games_left: int) -> float:
    """剩余局中「赢下的局数」的期望 —— 论文的奖励口径，这里只用于报告。"""
    return p * max(games_left, 0)


def series_equity(
    p: float, score_ours: int, score_opp: int, games_total: int,
    wins_needed: int, game_index: int,
) -> float:
    """把这个 p 一直维持下去，我方最终赢下系列赛的概率（含本局）。"""
    if score_ours >= wins_needed:
        return 1.0
    if score_opp >= wins_needed:
        return 0.0
    games_left = max(games_total - game_index + 1, 1)
    return clinch_probability(p, games_left, wins_needed - score_ours)


def series_swing(
    p: float, score_ours: int, score_opp: int, games_total: int,
    wins_needed: int, game_index: int,
) -> float:
    """这一局的分量 = 赢下它的晋级概率 − 输掉它的晋级概率，∈[0,1]。

    它回答的是「这一局值不值得拼命」，与「这套阵容多强」无关，只与比分和赛制有关。
    """
    if score_ours >= wins_needed or score_opp >= wins_needed:
        return 0.0
    left_after = max(games_total - game_index, 0)
    if score_ours + 1 >= wins_needed:
        win_branch = 1.0
    else:
        win_branch = clinch_probability(p, left_after, wins_needed - score_ours - 1)
    if score_opp + 1 >= wins_needed:
        lose_branch = 0.0
    else:
        lose_branch = clinch_probability(p, left_after, wins_needed - score_ours)
    return max(win_branch - lose_branch, 0.0)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def strength_offset(team_model, team_ours: str, team_opp: str, weight: float = 1.0) -> float:
    """对手强度基线的 logit 平移量：β · (我方强度 − 对方强度) · weight。

    β 直接取 M3 team 版模型里 `team_strength_diff` 的系数（+3.18），
    这样「加不加对手强度」在数学上是同一个模型的两个退化版本，而不是两套说法。
    weight=0 表示完全忽略对手（回到只看阵容）。
    """
    if not team_ours or not team_opp:
        raise BPCoachError("对手强度基线需要两支队名（team_ours / team_opp）")
    names = list(getattr(team_model, "feature_names", []) or [])
    if "team_strength_diff" not in names:
        raise BPCoachError(
            "这个胜率模型没有队伍强度列；请用 models/winrate_kpl2026_team.joblib"
        )
    beta = float(team_model.model.coef_[0][names.index("team_strength_diff")])
    table = (team_model.priors.team if team_model.priors else {}) or {}
    if team_ours not in table or team_opp not in table:
        missing = [t for t in (team_ours, team_opp) if t not in table]
        raise BPCoachError(f"队伍强度表里没有：{'、'.join(missing)}（队名要写全称）")
    return weight * beta * (table[team_ours] - table[team_opp])


# ====================== 配置 ======================
@dataclass
class RewardConfig:
    """奖励口径。

    mode:
        "series"  晋级概率（默认）。可跨局、跨比分比较，数值可直接读。
        "game"    单局胜率。做「这套阵容本身多强」的对比时更纯粹。
        "games"   剩余局胜场期望（论文口径）。注意它与 "game" 严格同序（差一个常数倍），
                  保留它只是为了能原样报告论文的那个数。
    team_offset : logit 平移，来自 strength_offset()；只改数值水平，不改同局排序。
    pool_weight : 英雄池影子价格的权重，会**改变排序**（唯一有这能力的旋钮），默认 0。
    denial      : 我方禁人步是否改用「拒绝价值」排序，默认开（见模块说明五）。
    denial_top  : 禁人步候选表里最多列几张（按拒绝价值取前 n）。
    """

    mode: str = "series"
    team_offset: float = 0.0
    pool_weight: float = 0.0
    denial: bool = True
    denial_top: int = 8

    def __post_init__(self) -> None:
        if self.mode not in ("series", "game", "games"):
            raise BPCoachError(f"未知奖励口径 {self.mode}，可选 series / game / games")
        if self.denial_top < 1:
            raise BPCoachError("denial_top 至少为 1")


@dataclass
class SearchConfig:
    """搜索预算与宽度。

    algorithm:
        "minimax" 等深全展开（默认）。叶子估值确定 + 宽度已剪到 6×4，深度 3 只需 ~240 个叶子，
                  价值是精确的，且不会因「谁被访问得多」而偏。
        "mcts"    UCT。要更长的战术线（max_depth=24）或要按预算截断时用。实测在补完
                  策略修好之后，它与精确解的候选排序 ρ≈0.89、偏差不显著（见模块说明四）；
                  早期的「UCT 把排序带偏」是补完策略糟糕的症状，不是 UCT 本身的问题。
    opp_risk:
        对方节点用「最坏情况」的权重，∈[0,1]。0 = 按 M2 概率加权（默认），
        1 = 纯最小（原教旨极小极大），中间 = 线性混合。为什么默认不是 1，见模块说明六。
    """

    algorithm: str = "minimax"
    depth: int = 3                # minimax 的层数（1 = 只看我方这一手 + 补完）
    simulations: int = 480        # 仅 mcts 用
    c_explore: float = 0.55       # UCT 探索常数（Q 在 [0,1] 尺度上）
    c_prior: float = 0.18         # M2 先验的加成系数，只影响同分时的先后
    top_ours: int = 6             # 我方展开宽度（M2 前 k）
    top_opp: int = 4              # 对手展开宽度
    max_depth: int = 24           # mcts 单次模拟最多往下走几手
    rollout: str = "m2"           # 叶子补完策略："m2"（真 M2 top-1，默认）/ "greedy"（手写近似）/ "prior"（按先验采样）/ "none"
    root_lane_cover: bool = True  # 根节点把「每条空路的最优卡」也放进候选（防 M2 漏招）
    opp_risk: float = 0.0         # 对方节点的最坏情况权重：0 = M2 加权期望，1 = 纯 min
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rollout not in ("m2", "greedy", "prior", "none"):
            raise BPCoachError(f"未知 rollout 策略 {self.rollout}")
        if self.algorithm not in ("minimax", "mcts"):
            raise BPCoachError(f"未知算法 {self.algorithm}，可选 minimax / mcts")
        if self.simulations <= 0:
            raise BPCoachError("simulations 必须为正")
        if self.depth < 1:
            raise BPCoachError("depth 至少为 1")
        if not 0.0 <= self.opp_risk <= 1.0:
            raise BPCoachError(f"opp_risk 必须在 [0,1] 之间（收到 {self.opp_risk}）")


# ====================== 输出 ======================
@dataclass
class Candidate:
    hero_id: int
    name: str
    value: float          # 平均奖励（按 RewardConfig.mode），**排序依据**（rank_by="value" 时）
    p_win: float          # rollout 平均单局胜率（搜索自己的口径）
    p_greedy: float       # 确定性参照：这一手之后双方按 M2 最可能走完的 p
    equity: float         # 按 p_greedy 折算的晋级概率（可复现、可跨局比较）
    visits: int
    policy: float         # M2 认为这一手会发生的概率
    share: float | None   # 搜索预算占比（visits / 总模拟数）；等深全展开下没有预算分配，为 None
    lane: str = ""
    lane_mix: list = field(default_factory=list)   # [(分路, 概率)]，标出摇摆位
    is_flex: bool = False
    delta_vs_best: float = 0.0
    denial: float = 0.0   # 禁人步：禁掉这张卡让对手上限掉的百分点（rank_by="denial" 时的排序依据）

    def as_dict(self) -> dict:
        return {
            "hero": self.name, "hero_id": self.hero_id,
            "value": round(self.value, 4), "p_win": round(self.p_win, 4),
            "p_greedy": round(self.p_greedy, 4),
            "equity": round(self.equity, 4), "visits": self.visits,
            "policy": round(self.policy, 4), "lane": self.lane,
            "is_flex": self.is_flex, "denial": round(self.denial, 5),
        }


@dataclass
class Recommendation:
    """一次推荐的完整结果：主推荐 + 备选 + 最优路径 + 对手应对。"""

    step_index: int
    total_steps: int
    is_our_turn: bool
    kind: str
    side: int
    our_side: int
    game_index: int
    score_ours: int
    score_opp: int
    candidates: list[Candidate]
    pv: list[dict]                       # 最优路径逐手
    replies: list[dict]                  # 对手最可能的应对（按 M2 概率排序，附 p 的变化）
    baseline_p: float                    # M2 贪心走完后的单局胜率（参照点）
    baseline_equity: float
    reward_mode: str
    n_sims: int
    n_evals: int                         # 叶子估值次数（M3 调用次数）：minimax=叶子数，mcts=模拟次数
    elapsed: float
    policy_name: str = "minimax"
    algorithm: str = "minimax"
    depth: int = 0
    degenerate: bool = False      # 排序依据在候选间完全没有区分度（见 _build 注释）
    rank_by: str = "value"        # "value"（搜索值）| "denial"（禁人步的拒绝价值）
    threat_before: float | None = None   # 禁人步：对手上限对应的 p（denial 的基线）
    denial_blank: int = 0         # 禁人步：禁了等于空过的合法卡数量（denial=0）
    opp_risk: float = 0.0         # 对手口径：0 = M2 加权期望，1 = 最坏情况（见模块说明六）

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def best_equity(self) -> float:
        return self.best.equity if self.best else self.baseline_equity

    def pv_prob(self) -> float:
        return self.pv[-1]["p_win"] if self.pv else self.baseline_p

    def pv_text(self) -> str:
        out = []
        for item in self.pv:
            who = "我方" if item["is_ours"] else "敌方"
            act = "ban" if item["kind"] == BAN else "选"
            out.append(f"{item['label']} {who}{act} {item['hero']}")
        return " → ".join(out)

    def as_dict(self) -> dict:
        return {
            "step": self.step_index + 1, "of": self.total_steps,
            "is_our_turn": self.is_our_turn, "kind": self.kind,
            "side": "blue" if self.side == BLUE else "red",
            "our_side": "blue" if self.our_side == BLUE else "red",
            "game_index": self.game_index,
            "score": [self.score_ours, self.score_opp],
            "reward_mode": self.reward_mode,
            "baseline_p": round(self.baseline_p, 4),
            "baseline_equity": round(self.baseline_equity, 4),
            "degenerate": self.degenerate,
            "rank_by": self.rank_by,
            "opp_risk": self.opp_risk,
            "threat_before": None if self.threat_before is None else round(self.threat_before, 4),
            "denial_blank": self.denial_blank,
            "best": self.best.as_dict() if self.best else None,
            "candidates": [c.as_dict() for c in self.candidates],
            "pv": self.pv,
            "replies": self.replies,
            "n_sims": self.n_sims, "n_evals": self.n_evals,
            "elapsed": round(self.elapsed, 3), "policy": self.policy_name,
            "algorithm": self.algorithm, "depth": self.depth,
        }


# ====================== 树节点 ======================
class _Node:
    __slots__ = ("state", "parent", "action", "prior", "children",
                 "n", "w", "p_sum", "terminal", "depth", "expanded", "ref_p")

    def __init__(self, state: DraftState, parent: "_Node | None" = None,
                 action: int | None = None, prior: float = 0.0, depth: int = 0) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: dict[int, _Node] = {}
        self.n = 0
        self.w = 0.0
        self.p_sum = 0.0
        self.terminal = state.is_finished
        self.depth = depth
        self.expanded = False
        self.ref_p: float | None = None     # 确定性参照：贪心补完后的 p（见 _build）

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0

    @property
    def p_mean(self) -> float:
        return self.p_sum / self.n if self.n else 0.0

    @property
    def is_ours(self) -> bool:
        step = self.state.current_step
        return True if step is None else step.side == self.state.our_side


# ====================== 搜索 ======================
class DraftSearch:
    """M4 主体。构造后无状态（每次 recommend 自带随机种子），可重复调用。"""

    def __init__(
        self,
        pool: HeroPool,
        markov,
        winrate,
        template: BPTemplate | None = None,
        config: SearchConfig | None = None,
        reward: RewardConfig | None = None,
    ) -> None:
        self.pool = pool
        self.m2 = markov
        self.m3 = winrate
        self.template = template or BPTemplate.load(
            getattr(markov, "template_id", None) or "kpl_2026_bo5_global"
        )
        self.cfg = config or SearchConfig()
        self.reward = reward or RewardConfig()
        self.priors = getattr(winrate, "priors", None)
        lm = pool.lane_model()
        self._lanes = tuple(lm.lanes)
        # 每张卡的「主路」下标：rollout 与根节点补路都用它，别每个候选重算 5 路排序
        self._main_lane = [
            max(range(len(self._lanes)), key=lambda i: lm.probs(h)[i])
            for h in range(pool.size)
        ]
        self._flex = [lm.is_flex(h) for h in range(pool.size)]
        # 频率先验（从 M2 的边际统计来）：rollout 采样用
        self._freq: dict[tuple[str, int], list[float]] = {}
        marg = getattr(markov, "marg", {}) or {}
        for kind in (BAN, PICK):
            for side in (BLUE, RED):
                vec = [float(v) for v in marg.get((kind, side), [0.0] * pool.size)]
                tot = sum(vec) or 1.0
                self._freq[(kind, side)] = [v / tot for v in vec]
        # 英雄池影子价格：这一手「消耗掉的强度」，用先验胜率相对基准的超出量表示
        base = getattr(self.priors, "base_win", 0.5) if self.priors else 0.5
        self._base = base
        self._win = list(getattr(self.priors, "win", None) or [base] * pool.size)
        self._price = [max(0.0, w - base) for w in self._win]

    # ---------------- 对外接口 ----------------
    def recommend(
        self,
        state: DraftState,
        simulations: int | None = None,
        depth: int | None = None,
        algorithm: str | None = None,
    ) -> Recommendation:
        """给出当前这一步的推荐（我方回合）或对手最凶的应对（对方回合）。

        algorithm: "minimax"（默认，等深全展开）| "mcts"（UCT，预算受限时用）
        """
        if state.is_finished:
            raise BPCoachError("B/P 已经走完，没有可推荐的下一手")
        t0 = time.time()
        algo = algorithm or self.cfg.algorithm
        root = _Node(state.clone())
        self._expand(root, is_root=True)
        if not root.children:
            raise BPCoachError("当前局面没有任何合法动作（检查模板与英雄池）")
        if algo == "mcts":
            n_sims = int(simulations or self.cfg.simulations)
            rng = random.Random(self.cfg.seed)
            for _ in range(n_sims):
                self._simulate(root, rng)
            # n_evals 统一记「叶子估值次数」：mcts 一次模拟估一个叶子，所以就是模拟次数
            return self._build(root, n_sims, n_sims, time.time() - t0,
                               policy_name="mcts", algorithm="mcts", depth=0)
        d = int(depth or self.cfg.depth)
        leaves = self._solve(root, d)
        return self._build(root, 0, leaves, time.time() - t0,
                           policy_name="minimax", algorithm="minimax", depth=d)

    def _solve(self, node: _Node, depth: int) -> int:
        """等深全展开。我方节点取最大，对方节点按 `opp_risk` 汇总（默认 M2 加权期望）。

        返回该子树评估过的叶子数。全展开（宽度已被 M2 剪到 6×4）所以没有
        「访问多的分支更乐观」的偏差，价值是精确的——精确于**这个汇总规则**。

        叶子为什么是「补完再打分」而不是直接给半成品阵容打分：
        半成品阵容的 M3 输出与成品的量纲不同，拿它比较等于拿半场比分判胜负。
        """
        if node.terminal:
            p = self._terminal_p(node.state)
            node.n, node.w, node.p_sum = 1, self._value(p, node.state), p
            return 1
        if depth <= 0:
            p = self._terminal_p(self._rollout(node.state, None))
            node.n, node.w, node.p_sum = 1, self._value(p, node.state), p
            return 1
        if not node.expanded:
            self._expand(node)
        if not node.children:
            p = self._terminal_p(self._rollout(node.state, None))
            node.n, node.w, node.p_sum = 1, self._value(p, node.state), p
            return 1

        vals: list[tuple[float, float, float]] = []      # (价值, p, M2 先验)
        leaves = 0
        for child in node.children.values():
            leaves += self._solve(child, depth - 1)
            vals.append((child.q, child.p_mean, child.prior))
        v, p = self._combine(vals, node.is_ours)
        node.n, node.w, node.p_sum = leaves, v * leaves, p * leaves
        return leaves

    def _combine(self, vals: list[tuple[float, float, float]],
                 ours: bool) -> tuple[float, float]:
        """把一个节点的子节点汇总成一个 (价值, p)。

        我方节点没有选择：取最大，这是搜索的全部意义。
        对方节点按 `opp_risk` 混合两种口径（见模块说明六）：
            期望  E = Σ prior_i·v_i / Σ prior_i     （对手按 M2 会怎么打）
            最坏  W = min_i v_i                      （对手专挑最狠的）
            v = (1−risk)·E + risk·W
        用 prior 当权重而不是均匀权重：M2 说「他大概会这么走」，不是「他随便走」。
        两边都用同一套权重算 p，避免「值按期望、p 按最坏」这种自相矛盾的表。
        """
        if ours:
            q, p, _ = max(vals, key=lambda x: x[0])
            return q, p
        risk = min(max(self.cfg.opp_risk, 0.0), 1.0)
        wq, wp, _ = min(vals, key=lambda x: x[0])
        if risk >= 1.0:
            return wq, wp
        tot = sum(x[2] for x in vals) or 1.0
        eq = sum(x[0] * x[2] for x in vals) / tot
        ep = sum(x[1] * x[2] for x in vals) / tot
        if risk <= 0.0:
            return eq, ep
        return (1.0 - risk) * eq + risk * wq, (1.0 - risk) * ep + risk * wp

    def _pv_child(self, node: _Node) -> "_Node | None":
        """最优路径的下一手：与 `_combine` 保持同一套口径，别让路径讲另一套故事。

        我方：值最大的那手。
        对方：最坏口径下取最伤我方的（路径要展示「他一定会这么打」）；
              期望口径下取**最可能**的那手（`prior` 最大）——期望值不对应某一个具体应法，
              硬拿它去挑一手会得到「一个谁都不会走、只是平均下来最糟」的动作。
        """
        if not node.children:
            return None
        if node.is_ours:
            return max(node.children.values(), key=lambda c: c.q)
        if self.cfg.opp_risk >= 1.0:
            return min(node.children.values(), key=lambda c: c.q)
        return max(node.children.values(), key=lambda c: (c.prior, -c.q))

    def myopic(self, state: DraftState, k: int | None = None) -> Recommendation:
        """贪心基线：只在 M2 前 k 手里挑「贪心补完后 M3 最高」的那一手。

        这是 M4 最该被打败的对手 —— 它也有 M3，只是不做树搜索、不考虑对手应对。
        """
        if state.is_finished:
            raise BPCoachError("B/P 已经走完，没有可推荐的下一手")
        t0 = time.time()
        k = k or self.cfg.top_ours
        pred = self.m2.predict(state, top_k=k, exclude_wasted=True)
        rows: list[tuple[int, float, float, float]] = []
        for hero, prob in pred.ranked[:k]:
            nxt = state.clone()
            nxt.apply(hero)
            final = self._greedy_complete(nxt)
            p = self._terminal_p(final)
            rows.append((hero, prob, p, self._value(p, final)))
        root = _Node(state.clone())
        for h, prob, p, v in rows:
            child_state = state.clone()
            child_state.apply(h)
            child = _Node(child_state, root, h, prob)
            child.n = 1
            child.w = v
            child.p_sum = p
            child.ref_p = p          # 贪心基线本来就只算这一个数，别让 _build 再算一遍
            root.children[h] = child
        return self._build(root, 0, 1, time.time() - t0, policy_name="myopic")

    # ---------------- 搜索核心 ----------------
    def _simulate(self, root: _Node, rng: random.Random) -> int:
        node = root
        path = [node]
        created = 0
        while node.children and not node.terminal and len(path) < self.cfg.max_depth:
            node = self._select(node)
            path.append(node)
        if not node.terminal and not node.expanded and len(path) < self.cfg.max_depth:
            created = self._expand(node)
        p = self._evaluate(node, rng)
        v = self._value(p, node.state)
        for nd in path:
            nd.n += 1
            nd.w += v
            nd.p_sum += p
        return created

    def _select(self, node: _Node) -> _Node:
        """UCT + M2 先验加成。

        我方节点看高的分支；对方节点只在**最坏情况**口径（opp_risk=1）下翻符号——
        那时「我们觉得好」等于「对手觉得糟」，探索最狠的一条才对。默认的期望口径下
        子节点值已经是他各种应法的加权平均，再翻一次符号就成了双重对抗，实测会把
        预算全压在对手最不可能走的分支上。
        未访问的子节点先各走一次，保证每个候选至少有一个样本——
        否则候选表里会出现「0 访问 = 0 价值」的假象。
        """
        cfg = self.cfg
        log_n = math.log(max(node.n, 1))
        best_child, best_score = None, -1e18
        flip = (not node.is_ours) and cfg.opp_risk >= 1.0
        for child in node.children.values():
            if child.n == 0:
                return child
            q = 1.0 - child.q if flip else child.q
            score = q + cfg.c_explore * math.sqrt(2 * log_n / child.n) + cfg.c_prior * child.prior
            if score > best_score:
                best_child, best_score = child, score
        return best_child or next(iter(node.children.values()))

    def _expand(self, node: _Node, is_root: bool = False) -> int:
        """展开一个节点：用 M2 的前 k 手当候选，根节点额外补「每条空路的最优卡」。"""
        node.expanded = True
        st = node.state
        if st.is_finished:
            node.terminal = True
            return 0
        side = st.side_to_act or st.our_side
        k = self.cfg.top_ours if side == st.our_side else self.cfg.top_opp
        pred = self.m2.predict(st, top_k=max(k, 1), exclude_wasted=True)
        if not pred.ranked:
            node.terminal = True
            return 0
        priors = dict(pred.ranked[:k])
        if is_root and self.cfg.root_lane_cover and st.kind_to_act == PICK:
            seen = set(priors)
            covered = {self._main_lane[h] for h in st.picks_of(side)}
            need = [i for i in range(len(self._lanes)) if i not in covered]
            for lane_idx in need:
                for hero, prob in pred.ranked:      # 已按 M2 概率降序
                    if hero in seen:
                        continue
                    if self._main_lane[hero] == lane_idx:
                        priors[hero] = max(prob, 0.02)
                        seen.add(hero)
                        break
        total = sum(priors.values()) or 1.0
        for hero, prob in priors.items():
            child_state = st.clone()
            try:
                child_state.apply(hero)
            except Exception:                       # 理论上不会发生；剪掉的非法动作直接跳过
                continue
            node.children[hero] = _Node(
                child_state, node, hero, prob / total, node.depth + 1
            )
        return 1 + len(node.children)

    def _evaluate(self, node: _Node, rng: random.Random) -> float:
        if node.terminal:
            return self._terminal_p(node.state)
        if self.cfg.rollout == "none":
            return self._terminal_p(node.state)
        return self._terminal_p(self._rollout(node.state, rng))

    def _terminal_p(self, state: DraftState) -> float:
        """M3 打分 + 对手强度平移。局面没走完也能算（用于 rollout 截断）。"""
        p = float(self.m3.predict_state(state).p_win)
        if self.reward.team_offset:
            p = _sigmoid(_logit(p) + self.reward.team_offset)
        return p

    def _rollout(self, state: DraftState, rng: random.Random | None) -> DraftState:
        """从当前局面补到 20 手结束。默认就是**真 M2 的 top-1**。

        为什么默认用确定性 argmax 而不是采样
            采样 rollout 的每个叶子是一个 0/1 附近的数，300 次模拟下单个候选的
            标准误约 8 个点，而候选之间的真实差距只有 1~3 个点——排序会退化成噪声排序。
            确定性补完后，同一节点重复模拟得到同一个值，候选之间的差就是真差。

        为什么从「手写近似」换成真 M2
            叶子估值 = M3(补完后的阵容)，所以补完策略就是这台搜索的「对手模型」。
            之前这里手写了一份近似（频率 + 强度 + 分路缺口），实测它与真 M2 的补完路径
            **12/12 个局面都在第一步就分歧**——频率项只依赖 (禁/选, 方)、完全不看棋盘，
            状态依赖只剩合法集合和分路缺口。也就是说叶子值衡量的其实是那个替代策略，
            而 `p_greedy` 用的是真 M2，两把尺子混在同一张表里。
            换掉的障碍一直是速度：M2.predict 单次 2.1ms × 20 手 = 40ms 一个叶子，
            深度 3 的 176 个叶子要 7 秒。markov.py 把打分向量化之后单次 0.18ms，
            一个叶子 2~3ms，深度 3 全程约 0.5s，这个障碍就没了。

        "greedy" 仍保留：它是**更凶的**对手（永远拿当下最强的卡，不管 M2 觉得他会不会），
        用来做敏感性对照——如果两种补完下推荐一致，那这个推荐就不依赖补完假设。
        """
        if self.cfg.rollout == "m2":
            return self._greedy_complete(state)
        st = state.clone()
        lm = self.pool.lane_model()
        deterministic = self.cfg.rollout != "prior"
        while not st.is_finished:
            legal = st.legal_hero_ids(exclude_wasted=True)
            if not legal:
                break
            step = st.current_step
            kind, side = step.kind, step.side
            freq = self._freq.get((kind, side))
            gaps = lm.gap(st.picks_of(side)) if kind == PICK else None
            scores = []
            for h in legal:
                s = 0.0
                if freq is not None:
                    s += 1.0 * math.log(freq[h] + 1e-4)
                if kind == PICK:
                    s += 2.0 * (self._win[h] - self._base)
                    if gaps is not None:
                        s += 3.0 * sum(p * g for p, g in zip(lm.probs(h), gaps))
                scores.append(s)
            if deterministic or rng is None:
                hero = legal[max(range(len(legal)), key=lambda i: scores[i])]
            else:
                m = max(scores)
                weights = [math.exp(3.0 * (s - m)) for s in scores]
                hero = rng.choices(legal, weights=weights, k=1)[0]
            st.apply(hero)
        return st

    def _greedy_complete(self, state: DraftState) -> DraftState:
        """照 M2 的最可能走法把剩下的手走完（baseline / 贪心基线共用）。"""
        st = state.clone()
        guard = 0
        while not st.is_finished and guard < 64:
            pred = self.m2.predict(st, top_k=1, exclude_wasted=True)
            if not pred.top_k:
                break
            st.apply(pred.top_k[0][0])
            guard += 1
        return st

    # ---------------- 拒绝价值（ban 步专用） ----------------
    def _lineup_p(self, state: DraftState, our_line: list[int],
                  opp_line: list[int]) -> float:
        """两套阵容在 M3 眼里的我方胜率（含对手强度平移），不经过 DraftState。

        为什么走 predict_lineup 而不是造一个假终局：造终局要处理「剩余 ban 用什么填」
        「共用选用名额不能撞」这些与阵容无关的事，撞上就得跳过候选，等于让实现细节
        污染价值函数。直接比较两套阵容没有这些麻烦，也快一倍。
        """
        score = state.series.score() if state.series else {BLUE: 0, RED: 0}
        ours, theirs = state.our_side, other(state.our_side)
        p = float(self.m3.predict_lineup(
            our_picks=list(our_line), opp_picks=list(opp_line),
            is_blue=(ours == BLUE), game_index=state.game_index,
            score_diff=score.get(ours, 0) - score.get(theirs, 0),
        ).p_win)
        if self.reward.team_offset:
            p = _sigmoid(_logit(p) + self.reward.team_offset)
        return p

    def _slot_beam(self, state: DraftState, line: list[int], slot: int,
                   taken: set[int], beam: int) -> list[int]:
        """第 slot 槽的替补候选（按「强度 + 补这一路」预筛）。

        预筛纯粹为了成本：全池 134 张 × 5 槽 × 2 轮 ≈ 1300 次 M3 调用，而一次 0.5ms。
        按 M3 自己的口径（强度 + 分路似然）取前 beam 个，漏掉真最优的概率很小；
        代价是 denial 会被高估一点点——因为 denial 取的是「替补里的最好那个」，
        少一个候选就等于少一个让对手少掉血的机会（见模块说明五的局限）。
        """
        lm = self.pool.lane_model()
        rest = [h for i, h in enumerate(line) if i != slot]
        gaps = lm.gap(rest)
        win, base = self._win, self._base
        block = set(rest) | taken
        scored = []
        for h in range(self.pool.size):
            if h in block:
                continue
            if self.pool.pick_excludes(h, block):
                continue        # 共用选用名额：不能跟场上已有的同组卡共存
            s = 2.0 * (win[h] - base) + 3.0 * sum(
                p * g for p, g in zip(lm.probs(h), gaps))
            scored.append((s, h))
        scored.sort(reverse=True)
        return [h for _, h in scored[:beam]]

    def _strongest_reply(
        self, state: DraftState, our_line: list[int],
        beam: int = 32, rounds: int = 2,
    ) -> tuple[list[int] | None, float | None]:
        """对手在这个局面下能凑出的**最强**五人，以及我方对应的 p。

        「最强」= 让 M3 判定的我方 p 最低的那套组合。先按「强度 + 分路缺口」铺一个种子，
        再逐槽做局部搜索。只用贪心铺一遍是不够的——这是第一版 denial 的 bug 所在：
        M3 除了先验胜率，还惩罚「摇摆位过多」（d_flex 系数为负）和「分路挤占」，
        贪心可能铺出一个先验很高、M3 却打低分的阵容，于是出现「禁掉其中一张，
        对手反而变强」的荒唐值（实测禁大乔让威胁 p 从 0.549 掉到 0.522）。
        爬坡之后，「他们的最强阵容」里不存在「拿掉会变强」的卡，denial 才非负。
        """
        lm = self.pool.lane_model()
        locked = list(state.picks_of(other(state.our_side)))
        taken = (set(our_line) | set(state.picks_of(BLUE)) | set(state.picks_of(RED))
                 | set(state.bans()))
        line = list(locked)
        used = set(taken)
        while len(line) < 5:
            gaps = lm.gap(line)
            best, best_s = None, -1e18
            for h in range(self.pool.size):
                if h in used or self.pool.pick_excludes(h, used):
                    continue
                s = 2.0 * (self._win[h] - self._base) + 3.0 * sum(
                    p * g for p, g in zip(lm.probs(h), gaps))
                if s > best_s:
                    best, best_s = h, s
            if best is None:
                return None, None
            line.append(best)
            used.add(best)
        p = self._lineup_p(state, our_line, line)
        for _ in range(max(rounds, 0)):
            improved = False
            for slot in range(len(locked), 5):
                rest = [h for i, h in enumerate(line) if i != slot]
                best_h, best_p = line[slot], p
                for h in self._slot_beam(state, line, slot, taken, beam):
                    q = self._lineup_p(state, our_line, rest + [h])
                    if q < best_p - 1e-9:
                        best_h, best_p = h, q
                if best_h != line[slot]:
                    line[slot] = best_h      # 原地换，别把顺序打乱（否则下一槽指错人）
                    p = best_p
                    improved = True
            if not improved:
                break
        return line, p

    def _denial(
        self, state: DraftState, beam: int = 32, rounds: int = 2,
    ) -> tuple[float | None, dict[int, float], list[int]]:
        """我方禁人步的拒绝价值。

        定义（方向容易写反，这里写死）：p 是**我方**胜率，
            denial(h) = p(对手失去 h 后) − p(对手最强时)  ≥ 0
        即「禁掉 h 之后我们的 p 能回升多少」= 「把他们的上限压掉多少」。

        我方阵容固定成「不禁人时的 M2 贪心走法」：只让对手的池子变，不让我方
        计划跟着变，差值才纯粹是「拒绝」，而不是双方一起顺移的噪声。
        """
        base_fin = self._greedy_complete(state)
        our_line = list(base_fin.picks_of(state.our_side))
        line, p_best = self._strongest_reply(state, our_line, beam, rounds)
        if line is None:
            return None, {}, our_line
        locked = set(state.picks_of(other(state.our_side)))
        base_taken = (set(our_line) | set(state.picks_of(BLUE))
                      | set(state.picks_of(RED)) | set(state.bans()))
        denial: dict[int, float] = {}
        for slot, hero in enumerate(line):
            if hero in locked:
                continue                      # 对手已经拿到手了，禁不了
            rest = [h for i, h in enumerate(line) if i != slot]
            # 禁掉 hero 之后，对手换成最好的替补：取「我方 p 回升最少」的那个，
            # 因为那个才是他们真正能保住的水平（对手会适应，不能假设他们不换）
            alt = min(
                (self._lineup_p(state, our_line, rest + [h])
                 for h in self._slot_beam(state, line, slot, base_taken | {hero}, beam)),
                default=p_best,
            )
            denial[hero] = max(0.0, alt - p_best)
        return p_best, denial, our_line

    # ---------------- 奖励 ----------------
    def _ctx(self, state: DraftState) -> tuple[int, int, int, int, int]:
        tpl = state.template
        score = state.series.score() if state.series else {BLUE: 0, RED: 0}
        our, opp = state.our_side, other(state.our_side)
        return (tpl.games, tpl.wins_needed, score.get(our, 0),
                score.get(opp, 0), state.game_index)

    def _value(self, p: float, state: DraftState) -> float:
        g, w, so, ss, gi = self._ctx(state)
        mode = self.reward.mode
        games_left = max(g - gi + 1, 1)
        if mode == "game":
            base = p
        elif mode == "games":
            base = expected_wins(p, games_left)
        else:
            base = series_equity(p, so, ss, g, w, gi)
        if self.reward.pool_weight:
            swing = series_swing(p, so, ss, g, w, gi)
            spent = sum(self._price[h] for h in state.picks_of(state.our_side))
            # (1 - swing)：这一局越不值钱，越该把强势英雄留给后面的局
            cost = self.reward.pool_weight * (1.0 - swing) * spent
            if mode == "games":
                # "games" 的奖励是「剩余局胜场」，量纲是 p × 剩余局数；
                # 早期这里被静默忽略（`and mode != "games"`），等于同一个旋钮
                # 在三种口径下有的生效有的不生效，而调用方看不出来。
                cost *= games_left
            base -= cost
        return base

    # ---------------- 结果组装 ----------------
    def _greedy_p(self, state: DraftState, hero: int | None = None) -> float:
        """确定性参照值：先落这一手，再让双方都照 M2 最可能走完，最后用 M3 打分。

        为什么每个候选都要这样一个数：搜索里的 p 是 rollout 的平均，不同候选的
        rollout 方差不同，直接拿去和 `baseline_p`（也是贪心走完）比就是拿两把尺子量。
        给出同口径的贪心 p，候选之间、候选与基线之间才真的可比。
        """
        st = state.clone()
        if hero is not None:
            st.apply(hero)
        return self._terminal_p(self._greedy_complete(st))

    def _build(self, root: _Node, n_sims: int, n_evals: int,
               elapsed: float, policy_name: str,
               algorithm: str = "minimax", depth: int = 0) -> Recommendation:
        st = root.state
        g, w, so, ss, gi = self._ctx(st)
        side = st.side_to_act or st.our_side

        # ---- 禁人步：排序依据换成「拒绝价值」 ----
        # 我方禁人不动我方那五个人（补完策略对双方是同一套），所以搜索值在候选间
        # 必然几乎同值——这不是搜索不准，是价值函数看不见「禁人」这件事。
        # 拒绝价值换一把尺子：禁掉这张卡，对手的天花板掉多少。
        rank_by = "value"
        threat_before: float | None = None
        denial: dict[int, float] = {}
        denial_blank = 0
        if (self.reward.denial and st.is_our_turn and st.kind_to_act == BAN
                and policy_name != "myopic"):
            threat_before, denial, _ = self._denial(st)
            # 只有落在「对手可达成的最强五人」里的卡，禁掉才会压到他们的上限；
            # 其余一律是 0，排序在它们之间没有意义，所以不列出来（只报个数）。
            alive = sum(1 for v in denial.values() if v > 1e-9)
            denial_blank = max(len(st.legal_hero_ids(exclude_wasted=True)) - alive, 0)
            if alive:
                rank_by = "denial"

        cands: list[Candidate] = []
        if rank_by == "denial":
            pol = dict(self.m2.predict(st, top_k=self.pool.size,
                                       exclude_wasted=True).ranked)
            top = [(h, d) for h, d in sorted(denial.items(), key=lambda kv: -kv[1])
                   if d > 1e-9][: self.reward.denial_top]
            for hero, d in top:
                child = root.children.get(hero)
                if child is not None and child.n:
                    p, q, visits = child.p_mean, child.q, child.n
                else:
                    st_after = st.clone()
                    try:
                        st_after.apply(hero)
                    except BPCoachError:
                        continue
                    p = self._greedy_p(st, hero)      # 本身已经是「禁完再贪心走完」的 p
                    q = self._value(p, st_after)
                    visits = 0
                cands.append(Candidate(
                    hero_id=hero, name=self.pool.name_of(hero),
                    value=q, p_win=p, p_greedy=p,
                    equity=series_equity(p, so, ss, g, w, gi),
                    visits=visits, policy=pol.get(hero, 0.0),
                    share=visits / n_sims if n_sims else None,
                    lane=self._lanes[self._main_lane[hero]],
                    lane_mix=self.pool.lane_model().top_lanes(hero, 2),
                    is_flex=self._flex[hero], denial=d,
                ))
            cands.sort(key=lambda c: -c.denial)
        else:
            for hero, child in root.children.items():
                if child.n:
                    p = child.p_mean
                    q = child.q
                else:
                    p = self._greedy_p(st, hero)
                    q = self._value(p, child.state)
                cands.append(Candidate(
                    hero_id=hero,
                    name=self.pool.name_of(hero),
                    value=q,
                    p_win=p,
                    p_greedy=p,
                    equity=series_equity(p, so, ss, g, w, gi),
                    visits=child.n,
                    policy=child.prior,
                    share=child.n / n_sims if n_sims else None,
                    lane=self._lanes[self._main_lane[hero]],
                    lane_mix=self.pool.lane_model().top_lanes(hero, 2),
                    is_flex=self._flex[hero],
                ))
            # 我方回合：值越大越好；对方回合：值越小越糟（对手会挑最伤我方的那手）
            cands.sort(key=lambda c: (-c.value if st.is_our_turn else c.value, -c.visits))
        # 每一行都给「确定性参照」：这手之后双方照 M2 走完的 p。
        # 早期为了控耗时只给前 6 名算，结果第 7 名以后的 p_greedy 其实是被
        # 直接赋成了搜索值（`p_greedy=p` 那一行），两个含义不同的数披同一个名字。
        # M2 向量化之后一次补完只要 2~3ms，没必要再省这一口。
        # 拒绝价值那一支不用重算：那里的 p 本来就是「禁完再照 M2 走完」的 p。
        if rank_by != "denial":
            for c in cands:
                child = root.children.get(c.hero_id)
                if child is not None and child.ref_p is None:
                    child.ref_p = self._greedy_p(st, c.hero_id)
                c.p_greedy = child.ref_p if child is not None and child.ref_p is not None \
                    else self._greedy_p(st, c.hero_id)
                c.equity = series_equity(c.p_greedy, so, ss, g, w, gi)
        for c in cands:
            c.delta_vs_best = c.value - cands[0].value
        # 退化：排序依据在候选间完全没有区分度。
        # 走 value 口径时 ban 步常见（禁人只是删选项，固定补完下不改变最终五个位置是谁）；
        # 走 denial 口径时若所有禁人都压不动对手上限，也会退化——那是「怎么禁都差不多」，
        # 同样不该被当成推荐。
        key = (lambda c: c.denial) if rank_by == "denial" else (lambda c: c.value)
        degenerate = len(cands) > 1 and (max(key(c) for c in cands) - min(key(c) for c in cands)) < 1e-6

        # 最优路径：第一手取**推荐的那一手**（否则会出现「推荐 A、路径是 B」的自相矛盾），
        # 之后 minimax 按「该谁行动」的汇总口径走（见 _pv_child），mcts 按访问数走。
        pv: list[dict] = []
        node = root
        first_hero = cands[0].hero_id if cands else None
        while node.children:
            if not pv and first_hero in node.children:
                child = node.children[first_hero]
            elif algorithm == "minimax":
                child = self._pv_child(node)
            else:
                child = max(node.children.values(), key=lambda c: (c.n, c.q))
            step = node.state.current_step
            if step is None:
                break
            pv.append({
                "step": step.index + 1,
                "label": step.order_in_phase,
                "side": "blue" if step.side == BLUE else "red",
                "is_ours": step.side == node.state.our_side,
                "kind": step.kind,
                "hero": self.pool.name_of(child.action),
                "p_win": round(child.p_mean, 4),
            })
            node = child
            if len(pv) >= max(self.cfg.max_depth, depth + 1):
                break

        # 对手应对：主推荐之后，对手换一种应法，我方 p 变成多少。
        # 只有「下一手真的是对手」时才列——KPL 的 p1-p1 是连续两手同一方，
        # 那种情况下列出来会把自己的第二手说成「对手的应对」。
        replies: list[dict] = []
        best = cands[0] if cands else None
        if best is not None:
            top = root.children.get(best.hero_id)
            nxt = top.state.current_step if top is not None else None
            if top is not None and nxt is not None and nxt.side != st.our_side:
                pred = self.m2.predict(top.state, top_k=3, exclude_wasted=True)
                for hero, prob in pred.top_k:
                    p = self._greedy_p(top.state, hero)
                    replies.append({
                        "hero": self.pool.name_of(hero),
                        "kind": nxt.kind,
                        "p_win": round(p, 4),
                        "equity": round(series_equity(p, so, ss, g, w, gi), 4),
                        "policy": round(prob, 4),
                        "delta": round(p - best.p_greedy, 4),
                    })

        base_p = self._greedy_p(st)
        return Recommendation(
            step_index=st.step_index, total_steps=st.total_steps,
            is_our_turn=st.is_our_turn, kind=st.kind_to_act or PICK, side=side,
            our_side=st.our_side, game_index=gi, score_ours=so, score_opp=ss,
            candidates=cands, pv=pv, replies=replies,
            baseline_p=base_p,
            baseline_equity=series_equity(base_p, so, ss, g, w, gi),
            reward_mode=self.reward.mode,
            n_sims=n_sims, n_evals=n_evals, elapsed=elapsed,
            policy_name=policy_name, algorithm=algorithm, depth=depth,
            degenerate=degenerate, rank_by=rank_by, threat_before=threat_before,
            denial_blank=denial_blank, opp_risk=self.cfg.opp_risk,
        )

    def __repr__(self) -> str:
        return (f"<DraftSearch sims={self.cfg.simulations} mode={self.reward.mode} "
                f"top={self.cfg.top_ours}/{self.cfg.top_opp}>")


# ====================== 便捷构造 ======================
def load(models_dir: str | pathlib.Path = "models", pool: HeroPool | None = None,
         **kwargs) -> DraftSearch:
    """从 models/ 目录装好 M2/M3 并返回一个 DraftSearch。

    默认用 `winrate_kpl2026_lineup.joblib`（只看阵容）：
    M4 是在「同一支队内部的边际优化」，把队伍强度混进搜索方向会跑偏。
    对手强度基线请显式用 `RewardConfig.team_offset` 传入，
    这样「算阵容」和「算这场比赛」两件事在代码里也是分开的。
    """
    from .markov import TransitionModel
    from .winrate import WinRateModel

    root = pathlib.Path(models_dir)
    if not root.is_absolute():
        root = pathlib.Path(__file__).resolve().parents[1] / root
    pool = pool or HeroPool.load()
    m2 = TransitionModel.load(root / "markov_kpl2026.json", pool)
    m3 = WinRateModel.load(root / "winrate_kpl2026_lineup.joblib", pool)
    return DraftSearch(pool, m2, m3, **kwargs)
