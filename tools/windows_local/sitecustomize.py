import logging
import os
import sys
import warnings


# Suppress the noisy Windows-only torch elastic redirect notice.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("deepspeed.runtime.config_utils").setLevel(logging.ERROR)
logging.getLogger("deepspeed.utils.logging").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress third-party deprecation chatter that does not affect packaged usage.
warnings.filterwarnings(
    "ignore",
    message="Using `TRANSFORMERS_CACHE` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*GPT2InferenceModel has generative capabilities.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Config parameter mp_size is deprecated.*",
)


if sys.platform == "win32":
    try:
        import contextlib
        from deepspeed.ops.op_builder.async_io import AsyncIOBuilder
        from deepspeed.ops.op_builder.builder import OpBuilder
        from deepspeed.ops.op_builder.gds import GDSBuilder

        AsyncIOBuilder.is_compatible = lambda self, verbose=False: False
        GDSBuilder.is_compatible = lambda self, verbose=False: False

        _orig_has_function = OpBuilder.has_function

        def _quiet_has_function(self, funcname, libraries, library_dirs=None, verbose=False):
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    return _orig_has_function(
                        self,
                        funcname,
                        libraries,
                        library_dirs=library_dirs,
                        verbose=False,
                    )

        OpBuilder.has_function = _quiet_has_function
    except Exception:
        pass
