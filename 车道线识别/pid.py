def _clamp(value, limits):
    # 限制值在指定的上下限之间
    lower, upper = limits  # 将limits元组解包为下限和上限
    if value is None:
        return None  # 如果值是None，则直接返回None
    elif (upper is not None) and (value > upper):
        return upper  # 如果上限存在且值大于上限，则返回上限值
    elif (lower is not None) and (value < lower):
        return lower  # 如果下限存在且值小于下限，则返回下限值
    return value  # 如果值在限制范围内，则直接返回该值

# P：比例  I：积分  D：微分

class PID(object):

    """PID 控制器"""

    def __init__(
            self,
            Kp=1.0,
            Ki=0.0,
            Kd=0.0,
            setpoint=0,
            sample_time=0.016666,
            output_limits=(None, None),
            auto_mode=True,
            proportional_on_measurement=False,
            differential_on_measurement=True,
            error_map=None,
            time_fn=None,
            starting_output=0.0,
    ):
        """
        初始化一个新的PID控制器。

        :param Kp: 比例增益Kp的值
        :param Ki: 积分增益Ki的值
        :param Kd: 微分增益Kd的值
        :param setpoint: PID尝试达到的初始设定点
        :param sample_time: 控制器生成新输出值之前应等待的时间（秒）。PID在连续调用时（例如在循环中）表现最佳，
                            但应设置采样时间，以便每次更新之间的时间差（接近）恒定。如果设置为None，则PID将在每次调用时计算新的输出值。
        :param output_limits: 要使用的初始输出限制，作为一个包含2个元素的迭代器，例如：(下限, 上限)。
                             输出永远不会低于下限或高于上限。任一限制也可以设置为None，以在该方向上没有限制。
                             设置输出限制还可以避免积分饱和，因为积分项永远不会被允许在限制之外增长。
        :param auto_mode: 控制器是否应启用（自动模式）或不启用（手动模式）
        :param proportional_on_measurement: 比例项是否应该直接在输入上计算，而不是在误差上（这是传统方式）。
                                            使用测量上的比例可以避免某些类型系统的超调。
        :param differential_on_measurement: 微分项是否应该直接在输入上计算，而不是在误差上（这是传统方式）。
        :param error_map: 用于将误差值转换为另一个约束值的函数。
        :param time_fn: 用于获取当前时间的函数，或者为None以使用默认值。这应该是一个不接受参数并返回表示当前时间的数字的函数。
                        默认值是尽可能使用time.monotonic()，否则使用time.time()。
        :param starting_output: PID输出的起始点。如果您开始控制一个已经处于设定点的系统，您可以设置这个参数为您对PID首次调用时应给出的输出的最佳猜测，
                               以避免PID输出零并将系统从设定点移开。
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self._min_output, self._max_output = None, None
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.error_map = error_map

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_time = None
        self._last_output = None
        self._last_error = None
        self._last_input = None

        if time_fn is not None:
            # 使用用户提供的 时间函数
            self.time_fn = time_fn
        else:
            import time

            try:
                # 获取单调时间以确保时间增量始终为正
                self.time_fn = time.monotonic
            except AttributeError:
                # time.monotonic() 不可用（使用python < 3.3），回退到 time.time()
                self.time_fn = time.time

        self.output_limits = output_limits
        self.reset()

        # 设置控制器的初始状态
        self._integral = _clamp(starting_output, output_limits)

    def __call__(self, input_, dt=None):
        """
        更新PID控制器。

        使用 *input_* 调用PID控制器，并在sample_time秒后计算并返回控制输出。
        如果没有计算新的输出，则返回之前的输出（如果还没有计算过值，则返回None）。

        :param dt: 如果设置，则使用此值作为时间步长，而不是实际时间。
                    这可以在模拟时使用，当模拟时间与实际时间不同时。
        """
        if not self.auto_mode:
            # 如果控制器不在自动模式，则返回最后的输出值
            return self._last_output

        now = self.time_fn()
        if dt is None:
            # 如果没有提供dt，则计算自上次更新以来的时间差
            dt = now - self._last_time if (now - self._last_time) else 1e-16
        elif dt <= 0:
            # 如果dt为非正数，则抛出异常
            raise ValueError('dt的值为负数或零，必须为正数: {}'.format(dt))

        if self.sample_time is not None and dt < self.sample_time and self._last_output is not None:
            # 如果未达到采样时间间隔，则返回最后的输出值
            return self._last_output

        # 计算误差项
        error = self.setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)
        d_error = error - (self._last_error if (self._last_error is not None) else error)

        # 检查是否需要对误差进行映射
        if self.error_map is not None:
            error = self.error_map(error)

        # 计算比例项
        if not self.proportional_on_measurement:
            # 常规的比例-误差计算，直接设置比例项
            self._proportional = self.Kp * error
        else:
            # 在测量上添加比例误差到误差和
            self._proportional -= self.Kp * d_input

        # 计算积分和微分项
        self._integral += self.Ki * error * dt
        self._integral = _clamp(self._integral, self.output_limits)  # 避免积分饱和

        if self.differential_on_measurement:
            self._derivative = -self.Kd * d_input / dt
        else:
            self._derivative = self.Kd * d_error / dt

        # 计算最终输出
        output = self._proportional + self._integral + self._derivative
        output = _clamp(output, self.output_limits)

        # 跟踪状态
        self._last_output = output
        self._last_input = input_
        self._last_error = error
        self._last_time = now

        return output

    def __repr__(self):
        """
        返回PID控制器的字符串表示形式，包含其关键参数的当前值。
        """
        return (
            '{self.__class__.__name__}('
            'Kp={self.Kp!r}, Ki={self.Ki!r}, Kd={self.Kd!r}, '
            'setpoint={self.setpoint!r}, sample_time={self.sample_time!r}, '
            'output_limits={self.output_limits!r}, auto_mode={self.auto_mode!r}, '
            'proportional_on_measurement={self.proportional_on_measurement!r}, '
            'differential_on_measurement={self.differential_on_measurement!r}, '
            'error_map={self.error_map!r}'
            ')'
        ).format(self=self)

    @property
    def components(self):
        """
        获取上一次计算的比例、积分和微分项作为单独的组件，以元组形式返回。
        这对于可视化控制器正在做什么或在调整难以调整的系统时非常有用。
        """
        return self._proportional, self._integral, self._derivative

    @property
    def tunings(self):
        """
        获取控制器使用的调参值，作为一个元组：(Kp, Ki, Kd)。
        """
        return self.Kp, self.Ki, self.Kd

    @tunings.setter
    def tunings(self, tunings):
        """
        设置PID控制器的调参值。

        参数:
        tunings -- 一个元组，包含比例(Kp)、积分(Ki)和微分(Kd)增益。
        """
        self.Kp, self.Ki, self.Kd = tunings

    @property
    def auto_mode(self):
        """
        获取控制器当前是否启用（自动模式）。
        """
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """
        启用或禁用PID控制器。

        参数:
        enabled -- 布尔值，表示是否启用自动模式。
        """
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        """
        启用或禁用PID控制器，并可选地设置最后的输出值。

        这在系统已经被手动控制，而PID需要接管时非常有用。
        在这种情况下，通过将自动模式设置为False来禁用PID，稍后当PID应该重新启动时，
        传递最后的输出变量（控制变量），它将被设置为PID设置为自动模式时的起始I项。

        :param enabled: 是否启用自动模式，True或False
        :param last_output: 当从手动模式切换到自动模式时，PID应该从这个最后的输出值（或控制变量）开始。
            如果PID已经在自动模式下，则此参数无效。
        """
        if enabled and not self._auto_mode:
            # 从手动模式切换到自动模式，重置PID控制器
            self.reset()

            # 设置积分项为最后的输出值，如果没有提供则默认为0
            self._integral = last_output if (last_output is not None) else 0
            # 使用clamp函数确保积分项在输出限制范围内
            self._integral = _clamp(self._integral, self.output_limits)

        # 设置PID控制器的自动模式状态
        self._auto_mode = enabled

    def reset(self):
        """
        重置PID控制器的内部状态。

        这将每个项设置为0，并清除积分项、最后输出和最后输入（用于导数计算）。
        """
        self._proportional = 0  # 将比例项重置为0
        self._integral = 0  # 将积分项重置为0
        self._derivative = 0  # 将导数项重置为0

        # 使用_clamp函数限制积分项在输出限制范围内
        self._integral = _clamp(self._integral, self.output_limits)

        # 重置时间跟踪变量
        self._last_time = self.time_fn()  # 使用时间函数获取当前时间作为最后时间
        self._last_output = None  # 清除最后的输出值
        self._last_input = None  # 清除最后的输入值
        self._last_error = None  # 清除最后的误差值


if __name__ == '__main__':
    pid = PID(1.0, 0, 0, 100, 0.11111)
    output = pid(10)