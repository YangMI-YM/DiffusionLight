from diffusers import UNet2DConditionModel

model = UNet2DConditionModel()

model.hey = "hey"

print("ConfigMixin property")
print(model.config)
print(hasattr(model, "config"))
print(getattr(model, "config"))
print(20 * "--")

print("ConfigMixin attribute")
print(model._internal_dict)
print(hasattr(model, "_internal_dict"))
print(getattr(model, "_internal_dict"))
print(20 * "--")

print("ModelMixin attribute")
print(model.hey)
print(hasattr(model, "hey"))
print(getattr(model, "hey"))

print("ConfigMixin function")
print(model.save_config)
print(hasattr(model, "save_config"))
print(getattr(model, "save_config"))
print(20 * "--")

print("ModelMixin function")
print(model.is_gradient_checkpointing)
print(hasattr(model, "is_gradient_checkpointing"))
print(getattr(model, "is_gradient_checkpointing"))
print(20 * "--")


print("Only config")
print(model.in_channels)
print(hasattr(model, "in_channels"))
print(getattr(model, "in_channels"))
print(20 * "--")

print("Good error message")
try:
    print(model.yes)
except Exception as e:
    print(e)
try:
    print(hasattr(model, "yes"))
except Exception as e:
    print(e)
try:
    print(getattr(model, "yes"))
except Exception as e:
    print(e)
print(20 * "--")