"""Library allowing a sh like infix syntax using pipes."""

__author__ = "Julien Palard <julien@python.org>"
__version__ = "2.2"
__credits__ = """Jérôme Schneider for teaching me the Python datamodel,
and all contributors."""

import builtins
import itertools
import functools
import os
import sys

from collections import deque
from contextlib import closing, suppress


REPR_EVALUATES = bool(getattr(sys, 'ps1', sys.flags.interactive))
if REPR_EVALUATES:
    sys.stderr.write('%s\n\n' % (
        "Running in the Python REPL: repr(Pipe) evaluates expressions."))


class Pipe:
    """
    Pipe class enable a sh like infix syntax.

    This class allows you to create a pipeline of operations by chaining functions
    together using the `|` operator. It wraps a function and its arguments, enabling
    you to apply the function to an input in a clean and readable manner.

    Examples
    --------
    Create a new Pipe operation:

        >>> from pipe import Pipe
        >>> @Pipe
        ... def double(iterable):
        ...     return (x * 2 for x in iterable)

    Use the Pipe operation:

        >>> result = [1, 2, 3] | double
        >>> list(result)
        [2, 4, 6]

    Notes
    -----
    ...

    """
    def __init__(self, function, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.input = []
        functools.update_wrapper(self, function)

    def __or__(self, other):
        with suppress(TypeError):
            if issubclass(other, Pipe):
                other = other()
            elif issubclass(other, object):
                if not (hasattr(other, 'extend') or hasattr(other, 'update')):
                    raise TypeError(
                        'Pipe closers must implement extend() or update()')
                other = other()
        if hasattr(other, 'describe_pipe'):
            other.describe_pipe(self.describe())
        if hasattr(other, 'extend'):
            other.extend(self)
            return other
        if hasattr(other, 'update'):
            other.update(self)
            return other
        if isinstance(other, Pipe):
            return other.__ror__(self)
        raise TypeError('Right-most element must be a Pipe, list, dict or set')

    def __ror__(self, other):
        """
        Implement the reverse pipe operator (`|`) for the object.

        Parameters
        ----------
        other : Any
            The left-hand operand of the `|` operator.

        Returns
        -------
        Any
            The result of applying the stored function to `other` with the
            provided arguments and keyword arguments.

        """
        dup = type(self)(self.function, *self.args, **self.kwargs)
        if isinstance(self.input, Pipe):
            dup.input = self.input.__ror__(other)
            return dup
        if self.input != []:
            raise ValueError('Pipeline does not support chaining from the left')
        dup.input = other
        return dup

    def guarantee_iterable(self, obj):
        if hasattr(obj, '__iter__'):
            return obj
        elif obj is None:
            return []
        return [obj]

    def __iter__(self):
        _input = self.guarantee_iterable(self.input)
        return iter(self.guarantee_iterable(
            self.function(_input, *self.args, **self.kwargs)))

    def __call__(self, *args, **kwargs):
        return type(self)(
            self.function,
            *self.args,
            *args,
            **self.kwargs,
            **kwargs,
        )

    def describe(self) -> str:
        def _desc(obj, _types=False):
            if hasattr(obj, 'describe'):
                return obj.describe()
            if _types:
                return type(obj).__name__
            return repr(obj)
        return "%s%s::<%s>(%s%s%s%s)" % (
            '%s | ' % _desc(self.input, _types=True) if self.input else '',
            type(self).__name__,
            self.function.__name__,
            ', '.join(_desc(a) for a in self.args),
            ', ' if (self.args and self.kwargs) else '',
            '**' if self.kwargs else '',
            '%s' % (self.kwargs,) if self.kwargs else '' ,
        )

    def __repr__(self) -> str:
        if REPR_EVALUATES:
            return '# %s\n%s' % (self.describe(), str(self))
        return self.describe()

    def __str__(self):
        return str(list(self))

    def __get__(self, instance, owner=None):
        return type(self)(
            function=self.function.__get__(instance, owner),
            *self.args,
            **self.kwargs,
        )


class TextPipe(Pipe):
    """A Pipe which emits lines of text."""
    def __str__(self):
        return '\n'.join('%s' % r for r in iter(self))

    def guarantee_iterable(self, obj):
        if isinstance(obj, bytes):
            obj = str(obj, 'utf-8')
        if isinstance(obj, str):
            return obj.splitlines()
        return super().guarantee_iterable(obj)


@Pipe
def take(iterable, qte):
    """Yield qte of elements in the given iterable."""
    if not qte:
        return
    for item in iterable:
        yield item
        qte -= 1
        if qte == 0:
            break


@Pipe
def tail(iterable, qte):
    """Yield qte of elements in the given iterable."""
    return deque(iterable, maxlen=qte)


@Pipe
def skip(iterable, qte):
    """Skip qte elements in the given iterable, then yield others."""
    for item in iterable:
        if qte == 0:
            yield item
        else:
            qte -= 1


@Pipe
def dedup(iterable, key=lambda x: x):
    """Only yield unique items. Use a set to keep track of duplicate data."""
    seen = set()
    for item in iterable:
        dupkey = key(item)
        if dupkey not in seen:
            seen.add(dupkey)
            yield item


@Pipe
def uniq(iterable, key=lambda x: x):
    """Deduplicate consecutive duplicate values."""
    iterator = iter(iterable)
    try:
        prev = next(iterator)
    except StopIteration:
        return
    yield prev
    prevkey = key(prev)
    for item in iterator:
        itemkey = key(item)
        if itemkey != prevkey:
            yield item
        prevkey = itemkey


enumerate = Pipe(builtins.enumerate)


@Pipe
def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    yield from itertools.permutations(iterable, r)


@Pipe
def netcat(to_send, host, port):
    """Send and receive bytes over TCP."""
    import socket
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.connect((host, port))
        for data in to_send | traverse:
            s.send(data)
        while 1:
            data = s.recv(4096)
            if not data:
                break
            yield data


@Pipe
def traverse(args):
    if isinstance(args, (bytes, str)):
        yield args
        return
    for arg in args:
        if isinstance(arg, (bytes, str)):
            yield arg
            continue
        try:
            yield from iter(arg) | traverse
        except TypeError:
            # not iterable --- output leaf
            yield arg


@Pipe
def tee(iterable):
    for item in iterable:
        sys.stdout.write(repr(item) + "\n")
        yield item


@Pipe
def select(iterable, selector):
    return builtins.map(selector, iterable)


map = select


@Pipe
def where(iterable, predicate):
    return (x for x in iterable if predicate(x))


filter = where


@Pipe
def take_while(iterable, predicate):
    return itertools.takewhile(predicate, iterable)


@Pipe
def skip_while(iterable, predicate):
    return itertools.dropwhile(predicate, iterable)


@Pipe
def groupby(iterable, keyfunc):
    return itertools.groupby(sorted(iterable, key=keyfunc), keyfunc)


@Pipe
def sort(iterable, key=None, reverse=False):  # pylint: disable=redefined-outer-name
    return sorted(iterable, key=key, reverse=reverse)


@Pipe
def reverse(iterable):
    try:
        return reversed(iterable)
    except TypeError:
        return reversed(list(iterable))


@Pipe
def t(iterable, y):
    if hasattr(iterable, "__iter__") and not isinstance(iterable, str):
        return iterable + type(iterable)([y])
    return [iterable, y]


@Pipe
def transpose(iterable):
    return list(zip(*iterable))


@Pipe
def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


chain = Pipe(itertools.chain.from_iterable)
chain_with = Pipe(itertools.chain)
islice = Pipe(itertools.islice)
izip = Pipe(zip)


class Sh:
    """Tools for creating pipes from Unix shell commands.

    Use `sh.COMMAND` to run any shell command as a pipe, or use
    `sh.make_pipes(globals())` to create global pipe commands from
    common shell utilities found on your system.

    Close pipes with `sh.Result`, `sh.Check` or `sh.Background` to
    generate lists of results where non-zero exit codes have been
    filtered out. Check will raise exceptions, otherwise non-zero
    codes will be on the result's `.exitcodes` attribute.
    """
    Cmd = None

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            for d in os.getenv('PATH').split(':'):
                if os.path.exists(os.path.join(d, name)):
                    sh_pipe = Sh.Cmd(name)
                    setattr(self, name, sh_pipe)
                    return sh_pipe
            raise

    def _shell_pipes(self, wanted):
        if wanted:
            whitelist = set(wanted)
        else:
            whitelist = {
                'awk', 'bash', 'bc', 'cal', 'cat', 'col', 'cut',
                'diff', 'du', 'df', 'echo', 'expand',
                'find', 'file', 'free', 'gpg', 'grep', 'head', 'hexdump',
                'ip', 'ls', 'netstat',
                'od', 'perl', 'printf', 'ps', 'pwd', 'sed', 'sh_sort',
                'shuf', 'sh_tail', 'unexpand', 'sh_uniq', 'uname', 'uptime',
                'sh_w', 'wc', 'whoami', 'xargs', 'xclip'}

        for d in os.getenv('PATH').split(':'):
            for f in os.listdir(d):
                for prefix in ('shell_', 'sh_'):
                    shell_f = prefix + f
                    if shell_f in whitelist:
                        whitelist.remove(shell_f)
                        yield (shell_f, Sh.Cmd(f))
                if f in whitelist:
                    whitelist.remove(f)
                    yield (f, Sh.Cmd(f))

        if wanted and whitelist:
            raise KeyError('Not found: %s' % ', '.join(whitelist))

    def make_pipes(self, target, *wanted):

        """
        Add shell pipes to the target namespace; either a custom list
        or a default set of common POSIX/GNU/Linux utilities.

        Examples:

            sh.make_pipes(globals())
            sh.make_pipes(globals(), 'uptime', 'ls', 'grep')

        Returns a list of functions added to the target.
        """
        added = []
        for fname, func in self._shell_pipes(wanted):
            if fname in target:
                raise KeyError('Cravenly refusing to override `%s`' % fname)
            target[fname] = func
            added.append(fname)
        return sorted(added)

    class Sh(TextPipe):
        pass

    class ExitCode:
        def __init__(self, code, pipe, cmd=None):
            self.exitcode = code
            if cmd:
                self.command = '%s(%s)' % (pipe, ('%s' % cmd)[1:-1])
            else:
                self.command = '(%s)' % (pipe,)

        def __repr__(self):
            return '<exitcode=%d from %s>' % (self.exitcode, self.command)

    class Result(list):
        def __init__(self, *args,
                _raise=False, bg=False, head=None, tail=None, discard=None):
            super().__init__(*args)

            self.bg = bg
            self.pipe = None
            self.exitcodes = []
            self._raise = RuntimeError if (_raise is True) else _raise
            self.child_pid = None
            self.killed = False
            self.seen = 0

            if bg and _raise:
                raise SystemError('Cannot raise on background errors')

            if tail is None and discard is None and head is None:
                discard = False
            elif discard is None and (head or tail):
                discard = True

            self.discard = discard
            self.head_max = head
            self.tail = deque([], tail) if tail else self

        finished = property(lambda s: bool(s.child_pid and s.exitcodes))
        running = property(lambda s: bool(s.child_pid and not s.finished))
        lines = property(lambda s: s[:])
        head = property(lambda s: s[:])

        def __str__(self):
            return super().__repr__()

        def __bool__(self):
            return not (self.exitcodes and self.exitcodes[-1])

        def describe_pipe(self, description):
            self.pipe = description

        def kill(self, _raise=False):
            if self.running:
                import signal
                self.killed = signal.SIGKILL if self.killed else signal.SIGINT
                os.kill(self.child_pid, self.killed)
                return self.killed
            elif _raise:
                raise OSError('Not running!')
            return False

        def _append(self, line):
            line = line.rstrip('\n')
            self.seen += 1

            if self.head_max is not None:
                if len(self) < self.head_max:
                    self.append(line)
            elif not self.discard:
                self.append(line)

            if self.tail is not self:
                self.tail.append(line)

        def collect(self, rfd):
            for line in rfd:
                self._append(line)
            self.join()

        def join(self):
            if self.child_pid:
                with suppress(ChildProcessError):
                    p, exitcode = os.waitpid(self.child_pid, 0)
                    self.exitcodes.append(
                        Sh.ExitCode(exitcode, self.pipe))

        def describe(self):
            details = 'head=%d%s seen=%d' % (
                len(self),
                ('' if self.tail is self else ' tail=%d' % len(self.tail)),
                self.seen)
            if self.exitcodes:
                details += ' exitcodes=' + ','.join(
                    str(e.exitcode) for e in self.exitcodes)

            if not self.bg:
                pass
            elif self.child_pid:
                details = 'child_pid=%s %s %s' % (
                    self.child_pid,
                    details,
                    'finished' if self.finished else 'running')
            else:
                details = 'parent_pid=%s' % os.getppid()
            return '(%s) %s' % (self.pipe, details)

        def __repr__(self):
            if REPR_EVALUATES:
                if self.tail is not self:
                    return '# %s\n%s' % (self.describe(), str(self.tail))
                return '# %s\n%s' % (self.describe(), str(self))
            return '<Sh.%s %s>' % (type(self).__name__, self.describe())

        def extend(self, vals):
            if self.bg:
                return self.bg_extend(vals)

            for v in vals:
                if isinstance(v, Sh.ExitCode):
                    if self._raise:
                        raise self._raise(v)
                    self.exitcodes.append(v)
                else:
                    self._append(v)

        def bg_extend(self, vals):
            rfd, wfd = os.pipe()
            rfd = os.fdopen(rfd, mode='r', encoding='utf-8')
            wfd = os.fdopen(wfd, mode='w', encoding='utf-8')

            kid_pid = os.fork()
            if kid_pid:
                import threading
                wfd.close()
                self.child_pid = kid_pid
                t = threading.Thread(target=self.collect, args=(rfd,), daemon=True)
                t.start()
                return

            rfd.close()
            exitcode = 0
            try:
                for v in vals:
                    if isinstance(v, Sh.ExitCode):
                        os._exit(v.exitcode)
                    wfd.write('%s\n' % v)
            except KeyboardInterrupt:
                exitcode = 1
            wfd.close()
            os._exit(exitcode)

    class Background(Result):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, bg=True, **kwargs)

    class Check(Result):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _raise=True, **kwargs)


@Sh.Sh
def Cmd(to_send, cmd=None, *args, _raise=True, **kwargs):
    """
    Create a shell Pipe for an external command, piping the input to
    the command on STDIN and yielding one line at a time from STDOUT
    back to the pipeline.

    Arguments are passed directly to the command. Keywords arguments
    are converted into `--key=val` and passed to the command as well.

    For all non-zero exit codes encountered, `Sh.ExitCode` objects
    will be yielded after the data stream is complete. Pipe to
    `sh.Result` (or `sh.Check` or `sh.Background`) to strip these
    off and store as attributes, or raise an exception:

        >>> 'hello' | sh.Cmd('grep', '-c', 'X')
        # str | Sh::<Cmd>('grep', '-c', 'X')
        0
        <exitcode=1 from Sh::<Cmd>('grep', '-c', 'X')>

        >>> rv = 'hello' | sh.Cmd('grep', '-c', 'X') |sh.Result
        >>> rv
        # (str | Sh::<Cmd>('grep', '-c', 'X')) head=1 seen=1 exitcodes=1
        ['0']
        >>> rv.exitcodes
        [<exitcode=1 from Sh::<Cmd>('grep', '-c', 'X')>]

        >>> 'hello' | sh.Cmd('grep', '-c', 'X') | sh.Check
        ...
        RuntimeError: <exitcode=1 from Sh::<Cmd>('grep', '-c', 'X')>

    Pipes can also run in the background:

        >>> tf = sh.Cmd('tail', '-f', '/var/log/dmesg') |sh.Background
        >>> tf
        <sh.Result (sh...) child_pid=1234 head=0 tail=0 seen=0 running>

    Note that because pipes run as generators, if you want to prevent
    execution if a command fails, you must collect the results into a
    list (or sh.result) first, before feeding it into the next stage
    of the pipeline.
    """
    import subprocess
    import threading

    if cmd is None:
        return

    eol = kwargs.pop('_eol', '\n')
    encoding = kwargs.pop('_encoding', 'utf-8')

    cmd = [cmd] + [str(a) for a in args]
    for key, val in kwargs.items():
        cmd.append('--%s=%s' % (key, val))

    child = subprocess.Popen(cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding=encoding,
        text=True)

    exitcodes = []
    def feed_child():
        with suppress(BrokenPipeError):
            for item in to_send:
                if isinstance(item, Sh.ExitCode):
                    exitcodes.append(item)
                else:
                    child.stdin.write('%s%s' % (item, eol))
            child.stdin.close()

    threading.Thread(target=feed_child, daemon=True).start()
    try:
        for result in child.stdout:
            yield result.rstrip('\n')

        for exitcode in exitcodes:
            yield exitcode

        exitcode = child.wait()
        if exitcode:
            yield Sh.ExitCode(exitcode, 'Sh::<Cmd>', cmd)
    except KeyboardInterrupt:
        child.kill()
        raise


sh = Sh()
Sh.Cmd = sh.Cmd = Cmd

if REPR_EVALUATES:
    sys.stderr.write('%s\n\n' % (
        Sh.__doc__.split('\n', 2)[-1].rstrip()))
