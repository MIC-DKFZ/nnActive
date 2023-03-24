import SimpleITK as sitk


def copy_geometry_sitk(target: sitk.Image, source: sitk.Image) -> sitk.Image:
    """Returns a version of target with origin, direction and spacing from source."""
    target.SetOrigin(source.GetOrigin())
    target.SetDirection(source.GetDirection())
    target.SetSpacing(source.GetSpacing())
    return target


def get_geometry_sitk(source: sitk.Image):
    out = {
        "origin": source.GetOrigin(),
        "direction": source.GetDirection(),
        "spacing": source.GetSpacing(),
    }
    return out


def set_geometry(target: sitk.Image, origin, direction, spacing):
    target.SetOrigin(origin)
    target.SetDirection(direction)
    target.SetSpacing(spacing)
    return target
