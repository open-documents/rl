
# register_to_config()

这个函数

```python
def register_to_config(self, **kwargs):
	# self.config_name: 类属性
	if self.config_name is None:
		raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
	# Special case for `kwargs` used in deprecation warning added to schedulers
	# TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
	# or solve in a more general way.
	kwargs.pop("kwargs", None)
	if not hasattr(self, "_internal_dict"):
	    internal_dict = kwargs
	else:
	    previous_dict = dict(self._internal_dict)
	    internal_dict = {**self._internal_dict, **kwargs}
	    # logger.debug(f"Updating config from {previous_dict} to {internal_dict}")
	self._internal_dict = FrozenDict(internal_dict)
```


# 加载模型或scheduler的配置字典 -- loadConfig()

该函数的作用：从本地或huggingface中下载JSON配置文件，并将JSON配置文件转换为字典。

返回结果：元组
- 第一个元素 -- 配置文件，类型字典。
- 第二个元素（需要通过return_unused_kwargs参数指定True，默认False） -- 类型字典。去掉使用到的键对应的元素后的kwargs，使用到的键包含subfolder、revision、cache_dir、force_download、resume_download、proxies、use_auth_token、local_files_only、user_agent。
- 第三个元素（需要通过return_commit_hash参数指定True，默认False） -- 类型str或None。配置文件对应hash值。

上述元素返回时，往往是这样。
```python
config, kwargs, commit_hash = cls.load_config()
```


```python
@classmethod
def load_config(
	cls,
	pretrained_model_name_or_path: Union[str, os.PathLike],
	return_unused_kwargs=False,
	return_commit_hash=False,
	**kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

	############################### 下面从kwargs中得到的变量都用于从huggingface中下载文件 ###############################
	subfolder = kwargs.pop("subfolder", None)
	revision = kwargs.pop("revision", None)
	# DIFFUSERS_CACHE 从上到下(如果没有设置当前序号的环境变量, 则使用下一个序号的内容, 同时路径中所有的~都被展开): 
	# 1. 环境变量 HUGGINGFACE_HUB_CACHE
	# 2. 环境变量 HF_HOME/hub
	# 3. 环境变量 XDG_CACHE_HOME/huggingface/hub
	# 4. 家目录/.cache/huggingface/hub
	cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
	force_download = kwargs.pop("force_download", False)
	proxies = kwargs.pop("proxies", None)                         # Optional[Dict]
	resume_download = kwargs.pop("resume_download", False)
	use_auth_token = kwargs.pop("use_auth_token", None)
	local_files_only = kwargs.pop("local_files_only", False)
	user_agent = kwargs.pop("user_agent", {})
	user_agent = {**user_agent, "file_type": "config"}
	user_agent = http_user_agent(user_agent)
	# 仅从传递的kwargs中移除"mirror", 而没有使用到
	_ = kwargs.pop("mirror", None)

	############################### 上面从kwargs中得到的变量都用于从huggingface中下载配置文件 ###############################

	if cls.config_name is None:
		raise ValueError("...")

	pretrained_model_name_or_path = str(pretrained_model_name_or_path)
	# 如果pretrained_model_name_or_path是文件, 则直接作为配置文件使用
	if os.path.isfile(pretrained_model_name_or_path):
		config_file = pretrained_model_name_or_path
	# 如果pretrained_model_name_or_path是文件夹, 则作为配置文件所在的目录使用
	elif os.path.isdir(pretrained_model_name_or_path):
	    # 
		if os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.config_name)):
			# Load from a PyTorch checkpoint
			config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
		# 
		elif subfolder is not None and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)):
			config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
		else:
			raise EnvironmentError("...")
	else:
		try:
			# Load from URL or cache if already cached
			config_file = hf_hub_download(
				pretrained_model_name_or_path,
				filename=cls.config_name,
				cache_dir=cache_dir,
				force_download=force_download,
				proxies=proxies,
				resume_download=resume_download,
				local_files_only=local_files_only,
				use_auth_token=use_auth_token,
				user_agent=user_agent,
				subfolder=subfolder,
				revision=revision,
		)
		except RepositoryNotFoundError:
			raise EnvironmentError(
				f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
				" listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
				" token having permission to this repo with `use_auth_token` or log in with `huggingface-cli"
				" login`."
			)
		except RevisionNotFoundError:
			raise EnvironmentError(
				f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
				" this model name. Check the model page at"
				f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
			)
		except EntryNotFoundError:
			raise EnvironmentError(
				f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
			)
		except HTTPError as err:
			raise EnvironmentError(
				"There was a specific connection error when trying to load"
				f" {pretrained_model_name_or_path}:\n{err}"
			)
		except ValueError:
			raise EnvironmentError(
				f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
				f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
				f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
				" run the library in offline mode at"
				" 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
			)
		except EnvironmentError:
			raise EnvironmentError(
				f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
				"'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
				f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
				f"containing a {cls.config_name} file"
			)
	try:
		# 将JSON配置文件转换为字典
		config_dict = cls._dict_from_json_file(config_file)
		commit_hash = extract_commit_hash(config_file)
	except (json.JSONDecodeError, UnicodeDecodeError):
		raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")


	# 既不用返回kwargs中未使用的部分, 也不用返回配置文件对应的hash值, 则直接返回配置文件对应的字典
	if not (return_unused_kwargs or return_commit_hash):
		return config_dict

	outputs = (config_dict,)
	if return_unused_kwargs:  # 除了返回配置文件对应的字典, 同时返回kwargs中未使用的部分
	    outputs += (kwargs,)
	if return_commit_hash:    # 除了返回配置文件对应的字典, 同时返回配置文件对应的hash值
	    outputs += (commit_hash,)
	return outputs
```


# 使用配置字典实例化Python类实例对象 -- from_config()

该函数的功能：使用


```python
@classmethod
def from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs):
	# <===== TO BE REMOVED WITH DEPRECATION
	# TODO(Patrick) - make sure to remove the following lines when config=="model_path" is deprecated
	if "pretrained_model_name_or_path" in kwargs:
		config = kwargs.pop("pretrained_model_name_or_path")
	if config is None:
	    raise ValueError("Please make sure to provide a config as the first positional argument.")
	# ======>

	if not isinstance(config, dict):
	    deprecation_message = "It is deprecated to pass a pretrained model name or path to `from_config`."
	    if "Scheduler" in cls.__name__:
	        deprecation_message += (
	            f"If you were trying to load a scheduler, please use {cls}.from_pretrained(...) instead."
	            " Otherwise, please make sure to pass a configuration dictionary instead. This functionality will"
	            " be removed in v1.0.0."
	        )
	    elif "Model" in cls.__name__:
	        deprecation_message += (
	            f"If you were trying to load a model, please use {cls}.load_config(...) followed by"
	            f" {cls}.from_config(...) instead. Otherwise, please make sure to pass a configuration dictionary"
	            " instead. This functionality will be removed in v1.0.0."
	        )
	    deprecate("config-passed-as-path", "1.0.0", deprecation_message, standard_warn=False)
	    config, kwargs = cls.load_config(pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

	init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)
	# Allow dtype to be specified on initialization
	if "dtype" in unused_kwargs:
	    init_dict["dtype"] = unused_kwargs.pop("dtype")
	# add possible deprecated kwargs
	for deprecated_kwarg in cls._deprecated_kwargs:
	    if deprecated_kwarg in unused_kwargs:
	        init_dict[deprecated_kwarg] = unused_kwargs.pop(deprecated_kwarg)
	# Return model and optionally state and/or unused_kwargs
	# 实例化对象
	model = cls(**init_dict)
	# make sure to also save config parameters that might be used for compatible classes
	model.register_to_config(**hidden_dict)
	# add hidden kwargs of compatible classes to unused_kwargs
	unused_kwargs = {**unused_kwargs, **hidden_dict}
	if return_unused_kwargs:
	    return (model, unused_kwargs)
	else:
	    return model
```