import weakref


class MetaNodeRegistry(type):
    """Metaclass providing a Node registry and its ID"""

    def __init__(cls, name, bases, attrs):
        # __init__ creates a class as this is a metaclass.
        super(MetaNodeRegistry, cls).__init__(name, bases, attrs)
        # Initialize instance registry.
        cls._instances = weakref.WeakSet()

    def __call__(cls, *args, **kwargs):
        # __call__ creates an instance, namely it calls __init__ and
        # __new__ methods.
        inst = super(MetaNodeRegistry, cls).__call__(*args, **kwargs)

        # Store weak reference to instance.
        cls._instances.add(inst)

        return inst

    def get_instances(cls, recursive=False):
        """Get all Node instaces in the registry.

        If recursive=True, search subclasses recursively.
        """
        instances = list(cls._instances)
        if recursive:
            for child in cls.__subclasses__():
                instances += child.get_instances(recursive=recursive)

        return list(set(instances))

    def get_id(cls, recursive=False):
        """Get an ID for an insntace that is being created now on.

        If recursive=True, search subclasses recursively. This method
        supposed to be called in __init__ of a class with this
        metaclass.
        """
        return len(cls.get_instances(recursive=recursive))


class Node(object, metaclass=MetaNodeRegistry):
    """Base class for reservoir, readout layer, and so on"""

    _name: str
    _id: int
    default_name = "node"

    def __init__(self):
        self._id = self.__class__.get_id()
        self._name = f"{self.__class__.default_name}_{self._id}"

    @property
    def name(self) -> str:
        """Return the name assigned to the instance."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        """Set a new name to the instance"""
        self._name = new_name

    @property
    def id(self) -> int:
        """Return the ID assigned to the instance."""
        return self._id
