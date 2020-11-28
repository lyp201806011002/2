def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积函数的前向传播

    参数：
        A_prev - 上一层的激活输出矩阵，维度为(m, n_H_prev, n_W_prev, n_C_prev)，（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
        W - 权重矩阵，维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）
        b - 偏置矩阵，维度为(1, 1, 1, n_C)，（1,1,1,这一层的过滤器数量）
        hparameters - 包含了"stride"与 "pad"的超参数字典。

    返回：
        Z - 卷积输出，维度为(m, n_H, n_W, n_C)，（样本数，图像的高度，图像的宽度，过滤器数量）
        cache - 缓存了一些反向传播函数conv_backward()需要的一些数据
    """

    # 获取来自上一层数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取权重矩阵的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取超参数hparameters的值
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 计算卷积后的图像的宽度高度，参考上面的公式，使用int()来进行板除
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # 使用0来初始化卷积输出Z
    Z = np.zeros((m, n_H, n_W, n_C))

    # 通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # 遍历样本
        a_prev_pad = A_prev_pad[i]  # 选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 切片位置定位好了我们就把它取出来,需要注意的是我们是“穿透”取出来的，
                    # 自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 执行单步卷积
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

    # 数据处理完毕，验证数据格式是否正确
    assert (Z.shape == (m, n_H, n_W, n_C))

    # 存储一些缓存值，以便于反向传播使用
    cache = (A_prev, W, b, hparameters)

    return (Z, cache)
