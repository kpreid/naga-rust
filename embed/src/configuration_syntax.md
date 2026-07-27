# Configuration

The available configuration options are:

* `allow_unimplemented = true | false` (default: `false`):

  Whether to allow the translated code to panic on entering code that cannot be
  translated, rather than failing the entire translation.

* `explicit_types = true | false` (default: `true`):

  Whether the translated code contains explicit types when they could be omitted.

* `include_functions = true | false` (default: `true`):

  Whether the translated code includes functions.

  This may be disabled to produce an output containing only `struct`s and `const`s,
  and remove any requirement to specify a `global_struct` or `resource_struct`.   

* `public_items = true | false` (default: `false`):

  Whether translated items have `pub` visibility instead of private.
  
  This option applies to all functions or methods, and all fields of translated structs.

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

* `rule(condition => effect)` or<br>
  `rule(effect)`:

  Defines rules that modify how specific parts of the shader are translated to Rust.

  The available **conditions** are:

  * `function(function_name_here)`: matches a single function according to its name in the shader code.
  * `struct(StructNameHere)`: matches a single struct according to its name in the shader code.

  The available **effects** are:

  * `derive(DeriveMacroNameHere)`: Adds `#[derive(DeriveMacroNameHere)]` to the translated struct.
  * `inline()` | `inline(always)` | `inline(never)`: Adds `#[inline]` to the translated function.
