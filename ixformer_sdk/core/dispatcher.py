class Dispatcher(object):
    """
    create object by dispatcher to reuse object.
    """

    _dispatcher = dict()

    @classmethod
    def dispatcher(cls, *args, **kwargs):
        key = cls.dispatcher_key(*args, **kwargs)
        obj = cls._dispatcher.get(key, None)
        if obj is None:
            obj = cls(*args, **kwargs)
            cls._dispatcher[key] = obj

        return obj

    @classmethod
    def dispatcher_key(cls, *args, **kwargs):
        raise NotImplementedError()
