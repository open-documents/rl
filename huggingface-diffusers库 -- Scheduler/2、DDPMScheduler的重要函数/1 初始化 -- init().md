

```python
@register_to_config
def __init__(
    self,
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear",
    trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
    variance_type: str = "fixed_small",
    clip_sample: bool = True,
    prediction_type: str = "epsilon",
    thresholding: bool = False,
    dynamic_thresholding_ratio: float = 0.995,
    clip_sample_range: float = 1.0,
    sample_max_value: float = 1.0,
    timestep_spacing: str = "leading",
    steps_offset: int = 0,
):
    if trained_betas is not None:
        self.betas = torch.tensor(trained_betas, dtype=torch.float32)
    elif beta_schedule == "linear":
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
        # this schedule is very specific to the latent diffusion model.
        self.betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        )
    elif beta_schedule == "squaredcos_cap_v2":
        # Glide cosine schedule
        self.betas = betas_for_alpha_bar(num_train_timesteps)
    elif beta_schedule == "sigmoid":
        # GeoDiff sigmoid schedule
        betas = torch.linspace(-6, 6, num_train_timesteps)
        self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.one = torch.tensor(1.0)
    
    # standard deviation of the initial noise distribution
    self.init_noise_sigma = 1.0
    
    # setable values
    self.custom_timesteps = False
    self.num_inference_steps = None
    self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
    self.variance_type = variance_type
```


 -- --

```python
################################################ configuration.utils.py ################################################
def register_to_config(init):
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
    
        # Ignore private kwargs in the init.

        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}

        if not isinstance(self, ConfigMixin):

            raise RuntimeError(

                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "

                "not inherit from `ConfigMixin`."

            )

  

        ignore = getattr(self, "ignore_for_config", [])

        # Get positional arguments aligned with kwargs

        new_kwargs = {}

        signature = inspect.signature(init)

        parameters = {

            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore

        }

        for arg, name in zip(args, parameters.keys()):

            new_kwargs[name] = arg

  

        # Then add all kwargs

        new_kwargs.update(

            {

                k: init_kwargs.get(k, default)

                for k, default in parameters.items()

                if k not in ignore and k not in new_kwargs

            }

        )

  

        # Take note of the parameters that were not present in the loaded config

        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:

            new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))

  

        new_kwargs = {**config_init_kwargs, **new_kwargs}

        getattr(self, "register_to_config")(**new_kwargs)

        init(self, *args, **init_kwargs)

  

    return inner_init
```