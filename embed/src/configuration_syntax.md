The available configuration options are:

* `allow_unimplemented = true | false` (default: `false`):

  Whether to allow the generated code to panic on entering code that cannot be
  translated, rather than failing generation.

* `explicit_types = true | false` (default: `false`):

  Whether the generated code contains explicit types when they could be omitted.

* `public_items = true | false` (default: `false`):

  Whether generated items have `pub` visibility instead of private.
  
  This option applies to all functions or methods, and all fields of generated structs.

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
