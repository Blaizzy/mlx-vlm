# Contributing to MLX VLM

Below are some tips to port Vision LLMs available on Hugging Face to MLX.

Next, from this directory, do an editable install:

```shell
pip install -e .
```

Then check if the model has weights in the
[safetensors](https://huggingface.co/docs/safetensors/index) format. If not
[follow instructions](https://huggingface.co/spaces/safetensors/convert) to
convert it.

After that, add the model file to the
[`mlx_vlm/models`](https://github.com/Blaizzy/mlx-vlm/tree/main/src/models)
directory. You can see other examples there. We recommend starting from a model
that is similar to the model you are porting.

Make sure the name of the new model file is the same as the `model_type` in the
`config.json`, for example
[llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/config.json#L7).

To determine the model layer names, we suggest either:

- Refer to the Transformers implementation if you are familiar with the
  codebase.
- Load the model weights and check the weight names which will tell you about
  the model structure.
- Look at the names of the weights by inspecting `model.safetensors.index.json`
  in the Hugging Face repo.

Additionally, add a test for the new model type to the [model
tests](https://github.com/Blaizzy/mlx-vlm/tree/main/src/tests/test_models.py).

From the `src/` directory, you can run the tests with:

```shell
python -m unittest discover tests/
```

## Pull Requests

1. Fork and submit pull requests to the repo.
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests and at least one review.
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   This should install hooks for running `black` and `clang-format` to ensure
   consistent style for C++ and python code.

   You can also run the formatters manually as follows on individual files:

     ```bash
     clang-format -i file.cpp
     ```

     ```bash
     black file.py
     ```

     or,

     ```bash
     # single file
     pre-commit run --files file1.py

     # specific files
     pre-commit run --files file1.py file2.py
     ```

   or run `pre-commit run --all-files` to check all files in the repo.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to mlx-examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
