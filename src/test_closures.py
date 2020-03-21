from types import SimpleNamespace

def test_closures():
  """
  find out: can ipython hot-reload closures? what about functions that are attributes? what about methods?
  - closures are fixed once they are initialized
  - non-closure function (and function attributes) can be edited and will live-reload. Dunno about methods.
  also: how do closures capture variables? by ref or by value? does it depend on the type?
  - closures variable capture does depend on the type. immutable (atomic?) values like strings are captured by value, but
    a mutable variable like a list is captured by reference. But the reference is tricky! (see below, in ipython shell)

    d = test_closures()
    d.clo() # "string [1,2,3,4]"
    d.b[3] = 2
    d.clo() # "string [1,2,3,2]"
    d.b = [1,4,5]
    d.clo() # "string [1,2,3,2]"

    d.notclo(d.a,d.b) # ipython hot-reloads changes to `not_a_closure` source
    d.clo() # ipython ignores changes to `closure` source
  """
  
  d = SimpleNamespace()
  d.a = "string"
  d.b = [1,2,3,4]
  d.clo = mkclosure(d.a,d.b)
  d.notclo = not_a_closure
  return d

def not_a_closure(a,b):
  print(a,b)
  # print(a,b)

def mkclosure(a,b):
  def closure():
    print(a,b)
    print(a,b)
  return closure

