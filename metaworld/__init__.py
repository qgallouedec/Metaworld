"""Proposal for a simple, understandable MetaWorld API."""
import abc
import pickle
import random
from collections import OrderedDict
from typing import List, NamedTuple, Type

import gym
import numpy as np
from gym import spaces

import metaworld.envs.mujoco.env_dict as _env_dict

EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


class MetaWorldEnv:
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    def set_task(self, task: Task) -> None:
        """Set the task.

        Raises:
            ValueError: If task.env_name is different from the current task.

        """


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> "OrderedDict[EnvName, Type]":
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
_MT_OVERRIDE = dict(partially_observable=False)

_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    tasks = []
    for (env_name, args) in args_kwargs.items():
        assert len(args["args"]) == 0
        env_cls = classes[env_name]
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args["kwargs"].copy()
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        env.close()
        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            del kwargs["task_id"]
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
    if seed is not None:
        np.random.set_state(st0)
    return tasks


def _ml1_env_names():
    tasks = list(_env_dict.ML1_V2["train"])
    assert len(tasks) == 50
    return tasks


class ML1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if not env_name in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=seed)
        self._test_tasks = _make_tasks(
            self._test_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=(seed + 1 if seed is not None else seed)
        )


class MT1(Benchmark):

    ENV_NAMES = _ml1_env_names()

    def __init__(self, env_name, seed=None):
        super().__init__()
        if not env_name in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")
        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        self._train_ = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed)
        self._test_tasks = []


class ML10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML10_V2["train"]
        self._test_classes = _env_dict.ML10_V2["test"]
        train_kwargs = _env_dict.ml10_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed)
        test_kwargs = _env_dict.ml10_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed)


class ML45(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML45_V2["train"]
        self._test_classes = _env_dict.ML45_V2["test"]
        train_kwargs = _env_dict.ml45_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed)
        test_kwargs = _env_dict.ml45_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed)


class MT10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT10_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT10_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed)
        self._test_tasks = []


class MT50(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.MT50_V2
        self._test_classes = OrderedDict()
        train_kwargs = _env_dict.MT50_V2_ARGS_KWARGS
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed)
        self._test_tasks = []


class Assembly(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("assembly-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["assembly-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="assembly-v2",
    entry_point="metaworld:Assembly",
    max_episode_steps=500,
)


class Basketball(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("basketball-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["basketball-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="basketball-v2",
    entry_point="metaworld:Basketball",
    max_episode_steps=500,
)


class BinPicking(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("bin-picking-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["bin-picking-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="bin-picking-v2",
    entry_point="metaworld:BinPicking",
    max_episode_steps=500,
)


class BoxClose(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("box-close-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["box-close-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="box-close-v2",
    entry_point="metaworld:BoxClose",
    max_episode_steps=500,
)


class ButtonPressTopdown(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("button-press-topdown-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["button-press-topdown-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="button-press-topdown-v2",
    entry_point="metaworld:ButtonPressTopdown",
    max_episode_steps=500,
)


class ButtonPressTopdownWall(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("button-press-topdown-wall-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["button-press-topdown-wall-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="button-press-topdown-wall-v2",
    entry_point="metaworld:ButtonPressTopdownWall",
    max_episode_steps=500,
)


class ButtonPress(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("button-press-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["button-press-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="button-press-v2",
    entry_point="metaworld:ButtonPress",
    max_episode_steps=500,
)


class ButtonPressWall(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("button-press-wall-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["button-press-wall-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="button-press-wall-v2",
    entry_point="metaworld:ButtonPressWall",
    max_episode_steps=500,
)


class CoffeeButton(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("coffee-button-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["coffee-button-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="coffee-button-v2",
    entry_point="metaworld:CoffeeButton",
    max_episode_steps=500,
)


class CoffeePull(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("coffee-pull-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["coffee-pull-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="coffee-pull-v2",
    entry_point="metaworld:CoffeePull",
    max_episode_steps=500,
)


class CoffeePush(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("coffee-push-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["coffee-push-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="coffee-push-v2",
    entry_point="metaworld:CoffeePush",
    max_episode_steps=500,
)


class DialTurn(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("dial-turn-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["dial-turn-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="dial-turn-v2",
    entry_point="metaworld:DialTurn",
    max_episode_steps=500,
)


class Disassemble(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("disassemble-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["disassemble-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="disassemble-v2",
    entry_point="metaworld:Disassemble",
    max_episode_steps=500,
)


class DoorClose(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("door-close-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["door-close-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="door-close-v2",
    entry_point="metaworld:DoorClose",
    max_episode_steps=500,
)


class DoorLock(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("door-lock-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["door-lock-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="door-lock-v2",
    entry_point="metaworld:DoorLock",
    max_episode_steps=500,
)


class DoorOpen(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("door-open-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["door-open-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="door-open-v2",
    entry_point="metaworld:DoorOpen",
    max_episode_steps=500,
)


class DoorUnlock(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("door-unlock-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["door-unlock-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="door-unlock-v2",
    entry_point="metaworld:DoorUnlock",
    max_episode_steps=500,
)


class HandInsert(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("hand-insert-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["hand-insert-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="hand-insert-v2",
    entry_point="metaworld:HandInsert",
    max_episode_steps=500,
)


class DrawerClose(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("drawer-close-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["drawer-close-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="drawer-close-v2",
    entry_point="metaworld:DrawerClose",
    max_episode_steps=500,
)


class DrawerOpen(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("drawer-open-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["drawer-open-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="drawer-open-v2",
    entry_point="metaworld:DrawerOpen",
    max_episode_steps=500,
)


class FaucetOpen(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("faucet-open-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["faucet-open-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="faucet-open-v2",
    entry_point="metaworld:FaucetOpen",
    max_episode_steps=500,
)


class FaucetClose(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("faucet-close-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["faucet-close-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="faucet-close-v2",
    entry_point="metaworld:FaucetClose",
    max_episode_steps=500,
)


class Hammer(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("hammer-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["hammer-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="hammer-v2",
    entry_point="metaworld:Hammer",
    max_episode_steps=500,
)


class HandlePressSide(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("handle-press-side-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["handle-press-side-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="handle-press-side-v2",
    entry_point="metaworld:HandlePressSide",
    max_episode_steps=500,
)


class HandlePress(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("handle-press-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["handle-press-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="handle-press-v2",
    entry_point="metaworld:HandlePress",
    max_episode_steps=500,
)


class HandlePullSide(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("handle-pull-side-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["handle-pull-side-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="handle-pull-side-v2",
    entry_point="metaworld:HandlePullSide",
    max_episode_steps=500,
)


class HandlePull(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("handle-pull-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["handle-pull-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="handle-pull-v2",
    entry_point="metaworld:HandlePull",
    max_episode_steps=500,
)


class LeverPull(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("lever-pull-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["lever-pull-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="lever-pull-v2",
    entry_point="metaworld:LeverPull",
    max_episode_steps=500,
)


class PegInsertSide(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("peg-insert-side-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["peg-insert-side-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="peg-insert-side-v2",
    entry_point="metaworld:PegInsertSide",
    max_episode_steps=500,
)


class PickPlaceWall(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("pick-place-wall-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["pick-place-wall-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="pick-place-wall-v2",
    entry_point="metaworld:PickPlaceWall",
    max_episode_steps=500,
)


class PickOutOfHole(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("pick-out-of-hole-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["pick-out-of-hole-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="pick-out-of-hole-v2",
    entry_point="metaworld:PickOutOfHole",
    max_episode_steps=500,
)


class Reach(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("reach-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["reach-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="reach-v2",
    entry_point="metaworld:Reach",
    max_episode_steps=500,
)


class PushBack(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("push-back-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["push-back-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="push-back-v2",
    entry_point="metaworld:PushBack",
    max_episode_steps=500,
)


class Push(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("push-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["push-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="push-v2",
    entry_point="metaworld:Push",
    max_episode_steps=500,
)


class PickPlace(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("pick-place-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["pick-place-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="pick-place-v2",
    entry_point="metaworld:PickPlace",
    max_episode_steps=500,
)


class PlateSlide(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("plate-slide-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["plate-slide-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="plate-slide-v2",
    entry_point="metaworld:PlateSlide",
    max_episode_steps=500,
)


class PlateSlideSide(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("plate-slide-side-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["plate-slide-side-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="plate-slide-side-v2",
    entry_point="metaworld:PlateSlideSide",
    max_episode_steps=500,
)


class PlateSlideBack(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("plate-slide-back-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["plate-slide-back-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="plate-slide-back-v2",
    entry_point="metaworld:PlateSlideBack",
    max_episode_steps=500,
)


class PlateSlideBackSide(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("plate-slide-back-side-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["plate-slide-back-side-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="plate-slide-back-side-v2",
    entry_point="metaworld:PlateSlideBackSide",
    max_episode_steps=500,
)


class PegUnplugSide(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("peg-unplug-side-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["peg-unplug-side-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="peg-unplug-side-v2",
    entry_point="metaworld:PegUnplugSide",
    max_episode_steps=500,
)


class Soccer(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("soccer-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["soccer-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="soccer-v2",
    entry_point="metaworld:Soccer",
    max_episode_steps=500,
)


class StickPush(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("stick-push-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["stick-push-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="stick-push-v2",
    entry_point="metaworld:StickPush",
    max_episode_steps=500,
)


class StickPull(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("stick-pull-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["stick-pull-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="stick-pull-v2",
    entry_point="metaworld:StickPull",
    max_episode_steps=500,
)


class PushWall(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("push-wall-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["push-wall-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="push-wall-v2",
    entry_point="metaworld:PushWall",
    max_episode_steps=500,
)


class ReachWall(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("reach-wall-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["reach-wall-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="reach-wall-v2",
    entry_point="metaworld:ReachWall",
    max_episode_steps=500,
)


class ShelfPlace(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("shelf-place-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["shelf-place-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="shelf-place-v2",
    entry_point="metaworld:ShelfPlace",
    max_episode_steps=500,
)


class SweepInto(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("sweep-into-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["sweep-into-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="sweep-into-v2",
    entry_point="metaworld:SweepInto",
    max_episode_steps=500,
)


class Sweep(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("sweep-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["sweep-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="sweep-v2",
    entry_point="metaworld:Sweep",
    max_episode_steps=500,
)


class WindowOpen(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("window-open-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["window-open-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="window-open-v2",
    entry_point="metaworld:WindowOpen",
    max_episode_steps=500,
)


class WindowClose(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str = "human") -> None:
        self.ml1 = ML1("window-close-v2")  # Construct the benchmark, sampling tasks
        self.meta_env = self.ml1.train_classes["window-close-v2"]()  # Create an environment with task `assembly`
        self.observation_space = spaces.Box(-1, 1, self.meta_env.observation_space.shape, dtype=np.float32)
        self.action_space = self.meta_env.action_space
        self.meta_env._partially_observable = False
        self.metadata["video.frames_per_second"] = self.meta_env.metadata["video.frames_per_second"]
        self.render_mode = render_mode

    def reset(self, seed=None):
        task = random.choice(self.ml1.train_tasks)
        self.meta_env.set_task(task)  # Set task
        self.meta_env._partially_observable = False
        return self.meta_env.reset().astype(np.float32), {}  # Reset environment

    def step(self, action):
        observation, reward, terminated, info = self.meta_env.step(action)
        info["is_success"] = info["success"]
        truncated = False
        return observation.astype(np.float32), reward, truncated, terminated, info

    def render(self):
        return self.meta_env.render(self.render_mode)

    def close(self):
        self.meta_env.close()


gym.register(
    id="window-close-v2",
    entry_point="metaworld:WindowClose",
    max_episode_steps=500,
)
