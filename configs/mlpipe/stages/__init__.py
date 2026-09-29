from . import finetune, export, validate, package

STAGES = {
    "finetune": finetune.run,
    "export": export.run,
    "validate": validate.run,
    "package": package.run,
}