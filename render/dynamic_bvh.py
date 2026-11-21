import math

# 包围盒统一表示为 (min_x, min_y, min_z, max_x, max_y, max_z)。


def bounds_union(a, b):
    """合并两个包围盒，返回同时包含二者的最小 AABB。"""
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
        max(a[4], b[4]),
        max(a[5], b[5]),
    )


def node_is_leaf(count, max_leaf):
    """叶子判定：图元数不超过叶子容量即为叶子。"""
    return count <= max_leaf


def refit_threshold(overlap, triangle_count):
    """refit 或 rebuild。

    决策只看包围盒重叠度（重叠度高说明 refit 后层次结构仍然退化，
    需要重建），与三角形数量无关；传入 triangle_count 仅为兼容调用方。
    """
    del triangle_count  # 刻意不参与判定
    return "rebuild" if overlap >= 0.5 else "refit"


def update_ancestors(path):
    """祖先路径去重：按从根到叶的顺序返回唯一的祖先节点。"""
    return sorted(set(path))


def instance_mask(material_mask, layer_mask):
    """实例掩码：材质层与图层都必须允许，按位取交集。"""
    return material_mask & layer_mask


def traversal_stack_limit(depth, base):
    """遍历栈容量：按树深度线性扩容，避免固定 32 层溢出回绕。"""
    return depth * base


def node_pool_reclaim(used, freed):
    """节点池回收：从占用数中扣除释放的空洞，剩余为有效占用。"""
    return max(0, used - freed)


def blas_double_buffer(active, next_buffer):
    """BLAS 双缓冲切换：返回新的 (active, next) 对。

    本帧写入 next_buffer，下一帧它成为 active，旧的 active 转为可覆写的
    next，从而避免顶点缓存被下一帧构建时覆写。
    """
    return next_buffer, active


def fence_required(producer, consumer):
    """跨队列同步：生产者和消费者不在同一队列时必须等待 fence。"""
    return producer != consumer


def frame_align(bvh_frame, camera_frame):
    """BVH 与相机对齐：两者构建于同一帧才可用于本帧光线。"""
    return bvh_frame == camera_frame


def instance_key(instance_id, blas_id):
    """实例唯一键，供多相机共享同一份实例树时去重。"""
    return f"{instance_id}:{blas_id}"


def leaf_capacity(triangles, max_leaf):
    """叶子容量：容纳全部三角形所需的叶子节点数（向上取整）。"""
    if max_leaf <= 0:
        raise ValueError("max_leaf 必须为正数")
    return math.ceil(triangles / max_leaf)


def box_overlap(a, b):
    """包围盒相交：三轴区间均有重叠（含边界接触）即相交。"""
    return (
        a[0] <= b[3] and b[0] <= a[3]
        and a[1] <= b[4] and b[1] <= a[4]
        and a[2] <= b[5] and b[2] <= a[5]
    )


def refit_root(child_bounds):
    """根节点 refit：合并全部子节点包围盒。"""
    if not child_bounds:
        raise ValueError("child_bounds 不能为空")
    root = child_bounds[0]
    for box in child_bounds[1:]:
        root = bounds_union(root, box)
    return root


def tlas_rebuild_needed(added, removed, moved):
    """TLAS 重建判定：实例集合或变换有任何变化都需要同步重建并提交。"""
    return (added + removed + moved) > 0
