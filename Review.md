
# Review

* Logging is misconfigured. A handler which writes to stdout is literally 
  always a good idea.

* Main app: 
  * Misleading comment about stratification being applied.
  * Unnecessary second data split (validation set unused)

* The Dataset unnecessarily supports sampling.

* Pipeline architecture:
  * not composable
  * every combination of features has a hard-coded pipeline implementation
  * Open-closed is violated
  * DRY is violated
  * hard to read and grasp
  * `get_feature_pipeline` is not typed, and it is difficult (for the caller of the function) to
    see, which feature set names are valid (better use an Enum)
  
  You did *not* find the right abstraction here.

* Wavelength features are unused.

* What is the idea behind the exponentiating transformer?