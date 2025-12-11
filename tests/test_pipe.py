import pipe


def test_uniq():
    assert list(() | pipe.uniq) == []


def test_take_zero():
    assert list([1, 2, 3] | pipe.take(0)) == []


def test_take_one():
    assert list([1, 2, 3] | pipe.take(1)) == [1]


def test_empty_iterable():
    assert list([] | pipe.take(999)) == []


def test_traverse():
    assert list('foo' | pipe.traverse) == ['foo']
    assert list(['foo', ['bar']] | pipe.traverse) == ['foo', 'bar']


def test_reverse():
    assert list([1, 2, 3] | pipe.reverse) == [3, 2, 1]
    assert list(range(3) | pipe.reverse) == [2, 1, 0]


def test_aliasing():
    is_even = pipe.where(lambda x: not x % 2)

    assert list(range(10) | is_even) == [0, 2, 4, 6, 8]


def test_netcat():
    data = [
        b"HEAD / HTTP/1.0\r\n",
        b"Host: python.org\r\n",
        b"\r\n",
    ]
    response = ""
    for packet in data | pipe.netcat("python.org", 80):
        response += packet.decode("UTF-8")
    assert "HTTP" in response


def test_enumerate():
    data = [4, "abc", {"key": "value"}]
    expected = [(5, 4), (6, "abc"), (7, {"key": "value"})]
    assert list(data | pipe.enumerate(start=5)) == expected


def test_class_support_on_methods():
    class Factory:
        n = 10

        @pipe.Pipe
        def mul(self, iterable):
            return (x * self.n for x in iterable)

    assert list([1, 2, 3] | Factory().mul) == [10, 20, 30]


def test_class_support_on_static_methods():
    class TenFactory:
        @pipe.Pipe
        @staticmethod
        def mul(iterable):
            return (x * 10 for x in iterable)

    assert list([1, 2, 3] | TenFactory.mul) == [10, 20, 30]


def test_class_support_on_class_methods():
    class Factory:
        n = 10

        @pipe.Pipe
        @classmethod
        def mul(cls, iterable):
            return (x * cls.n for x in iterable)

    assert list([1, 2, 3] | Factory.mul) == [10, 20, 30]

    Factory.n = 2
    assert list([1, 2, 3] | Factory.mul) == [2, 4, 6]

    obj = Factory()
    assert list([1, 2, 3] | obj.mul) == [2, 4, 6]


def test_class_support_with_named_parameter():
    class Factory:
        @pipe.Pipe
        @staticmethod
        def mul(iterable, factor=None):
            return (x * factor for x in iterable)

    assert list([1, 2, 3] | Factory.mul(factor=5)) == [5, 10, 15]


def test_pipe_repr():
    @pipe.Pipe
    def sample_pipe(iterable):
        return (x * 2 for x in iterable)

    assert repr(sample_pipe) == "Pipe::<sample_pipe>()"

    @pipe.Pipe
    def sample_pipe_with_args(iterable, factor):
        return (x * factor for x in iterable)

    pipe_instance = sample_pipe_with_args(3)
    real_repr = repr(pipe_instance)
    assert "Pipe::<sample_pipe_with_args>(" in real_repr
    assert "3" in real_repr


def test_pipe_input_coercion():
    assert list(1 | pipe.sort) == [1]
    assert list(1 | pipe.reverse) == [1]


def test_pipe_output_coercion():
    test_data = [1, 'foo', False, 'bar']
    count_str = lambda i: sum(1 for e in i if isinstance(e, str))
    assert count_str(test_data) == 2
    assert list(test_data | pipe.Pipe(count_str)) == [2]


def test_pipe_closing_lists():
    assert ([1] | pipe.sort | []) == [1]
    assert ([1] | pipe.sort | list) == [1]


def test_pipe_closing_sets():
    assert ([1] | pipe.sort | set()) == set([1])
    assert ([1] | pipe.sort | set) == set([1])


def test_pipe_closing_dicts():
    assert (range(2) | pipe.enumerate | dict()) == {0: 0, 1: 1}
    assert (range(2) | pipe.enumerate | dict) == {0: 0, 1: 1}


def test_pipe_closing_object():
    class Collector:
        def __init__(self):
            self.source = None
            self.data = []
        def describe_pipe(self, description):
            self.source = description
        def extend(self, values):
            self.data.extend(values)

    collected = [3, 4, 2, 1] | pipe.sort | Collector

    # Introspection: the Collector knows where the data came from
    assert collected.source == 'list | Pipe::<sort>()'

    # Data was collected correctly?
    assert collected.data == [1, 2, 3, 4]


def test_pipe_composition():
    pipe1 = pipe.enumerate | pipe.reverse
    pipe2 = pipe.traverse | pipe.sort
    pipe12 = pipe1 | pipe2
    pipe21 = pipe2 | pipe1

    assert [3, 2, 1] | pipe1 | list == [(2, 1), (1, 2), (0, 3)]
    expected12 = [0, 1, 1, 2, 2, 3]
    assert expected12 == [(2, 1), (1, 2), (0, 3)] | pipe2 | list
    assert expected12 == [3, 2, 1] | pipe1 | pipe2 | list
    assert expected12 == [3, 2, 1] | pipe12 | list

    assert [3, 2, 1] | pipe2 | list == [1, 2, 3]
    expected21 = [(2, 3), (1, 2), (0, 1)]
    assert expected21 == [1, 2, 3] | pipe1 | list
    assert expected21 == [3, 2, 1] | pipe2 | pipe1 | list
    assert expected21 == [3, 2, 1] | pipe21 | list
