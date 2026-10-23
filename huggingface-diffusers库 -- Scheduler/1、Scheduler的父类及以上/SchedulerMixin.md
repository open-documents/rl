
SchedulerMixin与ConfigMixin一样，是所有schedulers的父类。主要提供了2个方法：
- from_pretrained() -- 从本地的json文件中，或从Hub仓库中初始化一个scheduler。


下面对from_pretrained()和save_pretrained()做介绍。

-- -- 

# from_pretrained()

该函数的功能：从本地或huggingface中加载配置字典（JSON配置文件转换的字典），然后使用配置字典实例化Scheduler对象。

从源码也不难看出，该函数本质上就是~ConfigMixin.load_config()和~ConfigMixin.from_config()两个函数的组合。

```python
################################################### SchedulerMixin ###################################################

@classmethod
def from_pretrained(
	cls,
	pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
	subfolder: Optional[str] = None,
	return_unused_kwargs=False,
	**kwargs,
):
	r"""
	Instantiate a scheduler from a pre-defined JSON configuration file in a local directory or Hub repository.
	Parameters:
	    pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
	        Can be either:
	            - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
	              the Hub.
	            - A path to a *directory* (for example `./my_model_directory`) containing the scheduler
	              configuration saved with [`~SchedulerMixin.save_pretrained`].
	    subfolder (`str`, *optional*):
	        The subfolder location of a model file within a larger model repository on the Hub or locally.
	    return_unused_kwargs (`bool`, *optional*, defaults to `False`):
	        Whether kwargs that are not consumed by the Python class should be returned or not.
	    cache_dir (`Union[str, os.PathLike]`, *optional*):
	        Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
	        is not used.
	    force_download (`bool`, *optional*, defaults to `False`):
	        Whether or not to force the (re-)download of the model weights and configuration files, overriding the
	        cached versions if they exist.
	    resume_download (`bool`, *optional*, defaults to `False`):
	        Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
	        incompletely downloaded files are deleted.
	    proxies (`Dict[str, str]`, *optional*):
	        A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
	        'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
	    output_loading_info(`bool`, *optional*, defaults to `False`):
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
	<Tip>
	To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
	`huggingface-cli login`. You can also activate the special
	["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
	firewalled environment.
	</Tip>
	"""

	# 调用ConfigMixin的类方法load_config, 因此继承SchedulerMixin的类还必须继承ConfigMixin
	config, kwargs, commit_hash = cls.load_config(
		pretrained_model_name_or_path=pretrained_model_name_or_path,
		subfolder=subfolder,
		return_unused_kwargs=True,
		return_commit_hash=True,
		**kwargs,

	# 调用ConfigMixin的类方法from_config, 因此继承SchedulerMixin的类还必须继承ConfigMixin
	return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
```

# save_pretrained()


```python
def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
	"""
	Save a scheduler configuration object to a directory so that it can be reloaded using the
	[`~SchedulerMixin.from_pretrained`] class method.
	Args:
	    save_directory (`str` or `os.PathLike`):
	        Directory where the configuration JSON file will be saved (will be created if it does not exist).
	    push_to_hub (`bool`, *optional*, defaults to `False`):
	        Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
	        repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
	        namespace).
	    kwargs (`Dict[str, Any]`, *optional*):
	        Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
	"""

	# 调用ConfigMixin的实例方法save_config, 因此继承SchedulerMixin的类还必须继承ConfigMixin
	self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
```

# 例子

在某些情况下，Linux无法访问huggingface，此时from_pretrained()无法正常工作，

```python
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
scheduler.save_pretrained("...")    # ...为自定义路径
```

下面是保存的JSON文件。
```json
{
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.23.1",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "clip_sample": true,
  "clip_sample_range": 1.0,
  "dynamic_thresholding_ratio": 0.995,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "sample_max_value": 1.0,
  "steps_offset": 0,
  "thresholding": false,
  "timestep_spacing": "leading",
  "trained_betas": null,
  "variance_type": "fixed_small"
}
```