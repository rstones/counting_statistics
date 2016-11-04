
class PhysicalityError(ValueError):
    
    def __init__(self, *args, **kwargs):
        ValueError.__init__(self, *args, **kwargs)