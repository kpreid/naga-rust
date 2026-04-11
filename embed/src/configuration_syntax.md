The available configuration options are:

* `global_struct = StructNameHere`:
  Allow declarations of private global variables, generate a struct with the given name to hold
  them, and make all functions methods of that struct.

  The struct has one constructor method, which is declared as either
  `const fn new()` or `const fn new(resources: &ResourceStructName)`
  depending on whether `resource_struct` is also set.
  If there are no parameters, then it also implements [`Default`].
* `resource_struct = StructNameHere`:
  Allow declarations of resources (uniforms), generate a struct with the given name to hold
  them, and make all functions methods of that struct if `global_struct` is not also set.
* TODO: Other configuration options are not implemented.
