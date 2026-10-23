
功能：使用预训练模型的配置实例化一个Pytorch模型实例。

返回模型的特点：
- 默认处于eval()模式，要训练模型，需要将模型重新设置为训练模式model.train()。
- dropout模组没有激活。（dropout modules are deactivated.）

-- --

下面介绍该方法的参数。

1）pretrained_model_name_or_path：str 
- 类型 str ，huggingface Hub上的模型id。
- 类型 os.PathLike，通过~ModelMixin.save_pretrained()保存预训练模型参数的目录。

2）




）<font color="FF5500">low_cpu_mem_usage</font>：bool。

有2个作用：（）
- 通过不初始化权重而是直接加载预训练的权重来加速模型加载。
- 加载模型过程中保证使用不超过模型size的CPU内存。

需要注意2点：
- 只在Pytorch1.9.0后支持，在之前的版本设置为True会报错。
- 需要配合accelerate模组使用，否则即使设置low_cpu_mem_usage为True，实际也会是False。


）<font color="FF5500">device_map</font>：`str` | `Dict[str, Union[int, str, torch.device]]`

设置`device_map="auto"`，能够让Accelerate模组自动计算最优的`device_map`。更加详细的信息见 [designing a device map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).

A map that specifies where each submodule should go. It doesn't need to be defined for each
parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
same device.

需要注意3点：
- 只在Pytorch1.9.0后支持，在之前的版本设置为True会报错。
- 需要配合accelerate模组使用，否则会报错。
- 指定device_map参数必须指定low_cpu_mem_usage参数。






max_memory (`Dict`, *optional*):
    A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
    each GPU and the available CPU RAM if unset.
offload_folder (`str` or `os.PathLike`, *optional*):

                The path to offload weights if `device_map` contains the value `"disk"`.



```python
@classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Parameters:
            cache_dir (`Union[str, os.PathLike]`, *optional*):

                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache

                is not used.

            torch_dtype (`str` or `torch.dtype`, *optional*):

                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the

                dtype is automatically derived from the model's weights.

            force_download (`bool`, *optional*, defaults to `False`):

                Whether or not to force the (re-)download of the model weights and configuration files, overriding the

                cached versions if they exist.

            resume_download (`bool`, *optional*, defaults to `False`):

                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any

                incompletely downloaded files are deleted.

            proxies (`Dict[str, str]`, *optional*):

                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',

                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.

            output_loading_info (`bool`, *optional*, defaults to `False`):

                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.

            local_files_only(`bool`, *optional*, defaults to `False`):

                Whether to only load local model weights and configuration files or not. If set to `True`, the model

                won't be downloaded from the Hub.

            use_auth_token (`str` or *bool*, *optional*):

                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from

                `diffusers-cli login` (stored in `~/.huggingface`) is used.

            revision (`str`, *optional*, defaults to `"main"`):

                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier

                allowed by Git.

            from_flax (`bool`, *optional*, defaults to `False`):

                Load the model weights from a Flax checkpoint save file.

            subfolder (`str`, *optional*, defaults to `""`):

                The subfolder location of a model file within a larger model repository on the Hub or locally.

            mirror (`str`, *optional*):

                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not

                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more

                information.



            offload_state_dict (`bool`, *optional*):

                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if

                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`

                when there is some disk offload.

            variant (`str`, *optional*):

                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when

                loading `from_flax`.

            use_safetensors (`bool`, *optional*, defaults to `None`):

                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the

                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`

                weights. If set to `False`, `safetensors` weights are not loaded.

  

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with

        `huggingface-cli login`. You can also activate the special

        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a

        firewalled environment.

        """

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False

        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

		############################ 对low_cpu_mem_usage和device_map参数的检查 -- 开始 ############################

		# low_cpu_mem_usage参数需要配合accelerate库一起使用
		if low_cpu_mem_usage and not is_accelerate_available():
			low_cpu_mem_usage = False

		# device_map参数需要配合accelerate库一起使用
		if device_map is not None and not is_accelerate_available():
			raise NotImplementedError("...")

		# device_map参数只在Pytorch1.9.0以上支持
		if device_map is not None and not is_torch_version(">=", "1.9.0"):
			raise NotImplementedError("...")

		# low_cpu_mem_usage参数只在Pytorch1.9.0以上支持
		if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
			raise NotImplementedError("...")
	
			# 指定device_map参数必须指定low_cpu_mem_usage参数
		if low_cpu_mem_usage is False and device_map is not None:
			raise ValueError("...")

		############################ 对low_cpu_mem_usage和device_map参数的检查 -- 结束 ############################

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config

        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            user_agent=user_agent,
            **kwargs,
        )

        # load model
        model_file = None
        
        if from_flax:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=FLAX_WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            
            model = cls.from_config(config, **unused_kwargs)

            # Convert the weights

            from .modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model
            model = load_flax_checkpoint_in_pytorch_model(model, model_file)

        else:
            if use_safetensors:
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )

                except IOError as e:
                    if not allow_pickle:
                        raise e
                    pass

            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )

  

            if low_cpu_mem_usage:
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **unused_kwargs)

                # if device_map is None, load the state dict and move the params from meta device to the cpu

                if device_map is None:

                    param_device = "cpu"

                    state_dict = load_state_dict(model_file, variant=variant)

                    model._convert_deprecated_attention_blocks(state_dict)

                    # move the params from meta device to cpu

                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())

                    if len(missing_keys) > 0:

                        raise ValueError(

                            f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"

                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"

                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"

                            " those weights or else make sure your checkpoint file is correct."

                        )

  

                    unexpected_keys = load_model_dict_into_meta(

                        model,

                        state_dict,

                        device=param_device,

                        dtype=torch_dtype,

                        model_name_or_path=pretrained_model_name_or_path,

                    )

  

                    if cls._keys_to_ignore_on_load_unexpected is not None:

                        for pat in cls._keys_to_ignore_on_load_unexpected:

                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

  

                    if len(unexpected_keys) > 0:

                        logger.warn(

                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"

                        )

  

                else:  # else let accelerate handle loading and dispatching.

                    # Load weights and dispatch according to the device_map

                    # by default the device_map is None and the weights are loaded on the CPU

                    try:

                        accelerate.load_checkpoint_and_dispatch(

                            model,

                            model_file,

                            device_map,

                            max_memory=max_memory,

                            offload_folder=offload_folder,

                            offload_state_dict=offload_state_dict,

                            dtype=torch_dtype,

                        )

                    except AttributeError as e:

                        # When using accelerate loading, we do not have the ability to load the state

                        # dict and rename the weight names manually. Additionally, accelerate skips

                        # torch loading conventions and directly writes into `module.{_buffers, _parameters}`

                        # (which look like they should be private variables?), so we can't use the standard hooks

                        # to rename parameters on load. We need to mimic the original weight names so the correct

                        # attributes are available. After we have loaded the weights, we convert the deprecated

                        # names to the new non-deprecated names. Then we _greatly encourage_ the user to convert

                        # the weights so we don't have to do this again.

  

                        if "'Attention' object has no attribute" in str(e):

                            logger.warn(

                                f"Taking `{str(e)}` while using `accelerate.load_checkpoint_and_dispatch` to mean {pretrained_model_name_or_path}"

                                " was saved with deprecated attention block weight names. We will load it with the deprecated attention block"

                                " names and convert them on the fly to the new attention block format. Please re-save the model after this conversion,"

                                " so we don't have to do the on the fly renaming in the future. If the model is from a hub checkpoint,"

                                " please also re-upload it or open a PR on the original repository."

                            )

                            model._temp_convert_self_to_deprecated_attention_blocks()

                            accelerate.load_checkpoint_and_dispatch(

                                model,

                                model_file,

                                device_map,

                                max_memory=max_memory,

                                offload_folder=offload_folder,

                                offload_state_dict=offload_state_dict,

                                dtype=torch_dtype,

                            )

                            model._undo_temp_convert_self_to_deprecated_attention_blocks()

                        else:

                            raise e

  

                loading_info = {

                    "missing_keys": [],

                    "unexpected_keys": [],

                    "mismatched_keys": [],

                    "error_msgs": [],

                }

            else:

                model = cls.from_config(config, **unused_kwargs)

  

                state_dict = load_state_dict(model_file, variant=variant)

                model._convert_deprecated_attention_blocks(state_dict)

  

                model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(

                    model,

                    state_dict,

                    model_file,

                    pretrained_model_name_or_path,

                    ignore_mismatched_sizes=ignore_mismatched_sizes,

                )

  

                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
  
        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

  

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:

            return model, loading_info

  

        return model
```

# 可能出现的问题

1）AttributeError: module ‘safetensors‘ has no attribute ‘torch‘

当使用from_pretrained()方法出现无法加载模型的问题时，定位到问题，出现在load_state_dict()方法中，出现AttributeError: module ‘safetensors‘ has no attribute ‘torch‘异常。

解决方法，到safetensors文件夹的__init__.py中添加如下代码。
```python
from . import torch
```