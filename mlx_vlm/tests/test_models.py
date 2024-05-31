"""

The `test_models` module provides a suite of unit tests for evaluating various components of machine learning vision and language models. It is implemented using Python's `unittest` framework and is structured to test different aspects of the models, including the language model, multi-modal projector, and the vision tower. The module tests for correct configurations, proper data type handling, and expected output shapes ensuring the models perform as expected under different scenarios. These tests are essential for quality assurance and are designed to ensure the integrity of the modeling components within the `mlx_vlm` library.

The module defines a `TestModels` class that inherits from `unittest.TestCase` and contains several methods to conduct individual tests:

1. `language_test1runner`: Validates the language model component, checking aspects such as model type, layer count, vocabulary size, numerical precision, and output shapes.

2. `mm_projector_test_runner`: Assesses the multi-modal projector, verifying the transformation of vision features to the text hidden size and data types used.

3. `vision_test_runner`: Examines the vision tower functionality, assessing the model type, hidden size, number of channels, image resolution handling, and feature layer outputs.

Specific test methods for models like `nanoLlava`, `llava`, `idefics2`, and `paligemma` invoke these runners with appropriate configurations. These tests instantiate models with given parameters and then use the runners to conduct assertions on the model's behavior, outputs, and data types, ensuring they match the expected results for given inputs.

The module also provides comprehensive testing for models with varying configurations through model-specific tests. Assertions cover the integrity of inputs and outputs, correct tensor shapes, data types during model inference, and alignment with predefined model configurations.

It is important to note that the actual machine learning model implementations and the `mlx` core utilities for transformations are assumed to be provided by the `mlx_vlm` libraries, which the test module uses to instantiate and evaluate the models.

By running these tests, developers and quality assurance personnel can confirm that changes to the `mlx_vlm` library maintain the expected functionality and performance of machine learning models, providing confidence in the robustness and correctness of the system.
"""

import unittest

import mlx.core as mx
from mlx.utils import tree_map


class TestModels(unittest.TestCase):
    """
    A suite of test cases for validating the functionality of various vision-language models.
    This test suite includes tests for language models, vision models, multimodal projectors, and the
    integration of these components within specific architectures such as nanoLlava, llava, idefics2,
    and paligemma. It leverages the unittest library to assert model properties, data types, and
    the correctness of model outputs against known expectations.

    Methods:
        language_test_runner:
             Validates the properties of a given language model, including
            its type, vocabulary size, number of layers, and output shapes and types across
            different precision formats (float32, float16).
        mm_projector_test_runner:
             Validates the properties of multimodal projectors, including
            output shapes and types according to the vision and text hidden sizes for different
            precision formats.
        vision_test_runner:
             Validates the properties of a given vision tower, including its
            type, hidden size, and the output shape of hidden states for specified layers.
        test_nanoLlava:
             Configures and validates the nanoLlava model using predefined
            configurations for text and vision components, ensuring correct integration and
            functionality.
        test_llava:
             Configures and validates the llava model with specific text and vision
            configurations, examining its components and overall behavior.
        test_idefics2:
             Configures and tests the idefics2 model, focusing on its language and
            vision components, and additionally examines the integration with a Perceiver model
            component.
        test_paligemma:
             Sets up and validates the paligemma model, checking the language model,
            multimodal projector, and vision tower against expected configurations and outputs.

    Note:
        The tests use MXNet as the underlying framework for model operations and tensor manipulations.
        In each test, the respective components of the models are instantiated and verified using a series
        of assertions that check for expected properties and output consistency. The tests are designed
        to be comprehensive and ensure that model components interact correctly and produce the expected
        results when provided with synthetic input data.

    """

    def language_test_runner(self, model, model_type, vocab_size, num_layers):
        """
        Tests the language model by verifying its type, number of layers, and output shapes under different data types.
        This function conducts a suite of tests on a given language model. It checks whether the model's type matches the expected type and whether the actual number of model layers is equal to the expected number. The function also performs tests using two different data types (single precision float and half precision float) to ensure that the model produces outputs of the correct shape and data type. Specifically, it checks output shapes after running a forward pass with short sequences. It finally checks that the output data type matches the initially set data type for the model's parameters.

        Args:
            self (unittest.TestCase):
                 The test case class instance from which this method is called.
            model (nn.Module):
                 The language model being tested.
            model_type (str):
                 Expected type of the model.
            vocab_size (int):
                 Expected vocabulary size of the model.
            num_layers (int):
                 Expected number of layers in the model.

        Raises:
            AssertionError:
                 If any of the tests fail, indicating that the model does not meet the expected specifications.

        """
        self.assertEqual(model.model_type, model_type)
        self.assertEqual(len(model.layers), num_layers)

        batch_size = 1

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs, cache = model(inputs)
            self.assertEqual(outputs.shape, (batch_size, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs, cache = model(
                mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache
            )
            self.assertEqual(outputs.shape, (batch_size, 1, vocab_size))
            self.assertEqual(outputs.dtype, t)

    def mm_projector_test_runner(
        self, mm_projector, vision_hidden_size, text_hidden_size
    ):
        """
        Runs tests on the multimodal projector to ensure correct shape and data type of outputs.
        This function runs a series of tests on the provided multimodal projector instance to check if it
        produces the correct output shape and maintains the correct data type after processing. It
        iterates over a predefined list of data types (float32 and float16) and verifies that the output
        from the multimodal projector matches the expected shape and type for each test case. The
        function primarily checks for consistency in the output shape based on the batch size and the
        text hidden size. Additionally, the data type of the outputs is also verified to match the input
        data type. If any of the checks fail, the test runner will raise an assertion error.

        Args:
            mm_projector (MultimodalProjector):
                 An instance of a MultimodalProjector, which is the
                subject of the tests.
            vision_hidden_size (int):
                 The size of the hidden layer for vision features in the multimodal
                projector.
            text_hidden_size (int):
                 The size of the hidden layer for text features expected in the output
                of the multimodal projector.

        Raises:
            AssertionError:
                 If the output shape or data type does not match the expected values.

        """
        batch_size = 1

        for t in [mx.float32, mx.float16]:
            mm_projector.update(
                tree_map(lambda p: p.astype(t), mm_projector.parameters())
            )

            vision_features = mx.random.uniform(
                shape=(batch_size, vision_hidden_size), dtype=t
            )
            input_tensor = mx.array(vision_features)

            outputs = mm_projector(input_tensor)
            self.assertEqual(outputs.shape, (batch_size, text_hidden_size))
            self.assertEqual(outputs.dtype, t)

    def vision_test_runner(
        self,
        vision_tower,
        model_type,
        vision_hidden_size,
        num_channels,
        image_size: tuple,
        vision_feature_layer=-2,
    ):
        """
        Performs a unit test for a given vision tower model.
        This function confirms that the vision tower model adheres to the expected specifications.
        It uses an assertEqual test to check if the model type is as expected and to validate the
        output tensor shape from the hidden states at a specified hidden layer.

        Args:
            vision_tower (VisionTower):
                 The vision tower model to be tested.
            model_type (str):
                 The expected type of the vision model.
            vision_hidden_size (int):
                 The expected size of the hidden state in the vision
                tower's feature layer.
            num_channels (int):
                 The number of channels in the input image.
            image_size (tuple):
                 The dimensions of the input image as a (height, width) tuple.
            vision_feature_layer (int, optional):
                 The index of the hidden layer from
                which the output tensor shape will be checked. Defaults to -2, typically
                representing the second to last layer.

        Raises:
            AssertionError:
                 If the actual model type does not match the expected type,
                or if the output tensor shape does not match the expected shape.

        """
        self.assertEqual(vision_tower.model_type, model_type)

        batch_size = 1

        input_tensor = mx.random.uniform(
            shape=(batch_size, image_size[0], image_size[1], num_channels)
        )

        # Perform a forward pass
        *_, hidden_states = vision_tower(input_tensor, output_hidden_states=True)
        # Check the output tensor shape
        self.assertEqual(
            hidden_states[vision_feature_layer][-1][-1].shape, (vision_hidden_size,)
        )

    def test_nanoLlava(self):
        """
        Tests the nanoLlava model's components, including the language model, multimodal projector, and vision tower.
        This method sets up the configurations for the text and vision models, then proceeds
        to instantiate the ModelConfig and nanoLlava Model. It uses predefined test runner
        methods to validate the behavior and functionality of the language model, multimodal
        projector, and vision tower components of the model.

        Raises:
            Assertions or errors from the individual test runners if any of the components do not
            perform as expected.

        Notes:
            The test runners should assert the correct dimensions and expected output for each
            component of the model. If any of the assertions fail, the test case will fail and
            the error will be raised.

        """
        from mlx_vlm.models import nanoLlava

        text_config = nanoLlava.TextConfig(
            model_type="qwen2",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = nanoLlava.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=384,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        args = nanoLlava.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava-qwen2",
            auto_map={
                "AutoConfig": "configuration_llava_qwen2.LlavaQwen2Config",
                "AutoModelForCausalLM": "modeling_llava_qwen2.LlavaQwen2ForCausalLM",
            },
            hidden_size=1024,
            mm_hidden_size=1152,
            mm_vision_tower="google/siglip-so400m-patch14-384",
            mm_projector_type="mlp2x_gelu",
            ignore_index=-100,
            image_token_index=-200,
            vocab_size=151936,
        )

        model = nanoLlava.Model(args)

        self.language_test_runner(
            model.language_model,
            args.text_config.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.mm_projector,
            args.vision_config.hidden_size,
            args.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower.vision_tower,
            args.vision_config.model_type,
            args.vision_config.hidden_size,
            args.vision_config.num_channels,
            (args.vision_config.image_size, args.vision_config.image_size),
        )

    def test_llava(self):
        """
        Tests various components of the LLAVA model including the language model, multi-modal projector, and vision model.
        This test function validates different aspects of the LLAVA model to ensure that they are functioning as expected. It checks the language model for compliance with the specified configuration parameters such as vocab_size and num_hidden_layers. Additionally, it tests the multi-modal projector to assert correct dimensionality transformations between the vision and language model's embeddings. Finally, it validates the vision tower's attributes like the model_type and hidden_size to ensure they match the expected values for the provided vision configuration.

        Args:
            self:
                 An instance of the unittest.TestCase or a similar class implementing testing functionality.

        Raises:
            AssertionError:
                 If any of the tests fail, indicating that a component of the LLAVA model
                does not meet the expected specifications.

        """
        from mlx_vlm.models import llava

        text_config = llava.TextConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=11008,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=32,
            rope_theta=10000.0,
            rope_traditional=False,
            rope_scaling=None,
        )

        vision_config = llava.VisionConfig(
            model_type="clip_vision_model",
            num_hidden_layers=23,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
            vocab_size=32000,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        args = llava.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="llava",
            ignore_index=-100,
            image_token_index=32000,
            vocab_size=32000,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava.Model(args)

        self.language_test_runner(
            model.language_model,
            args.text_config.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            args.vision_config.hidden_size,
            args.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            args.vision_config.model_type,
            args.vision_config.hidden_size,
            args.vision_config.num_channels,
            (args.vision_config.image_size, args.vision_config.image_size),
        )

    def test_idefics2(self):
        """
        Performs tests on the Idefics2 model including its language and vision components.
        This function is designed to execute testing procedures on the Idefics2 model which is
        a multimodal model with specific configurations for language and vision processing.
        It initializes a set of configurations for the text, vision, and perceiver modules,
        creates an instance of the model with those configurations, and runs tests on the
        language and vision models separately.
        The function takes no arguments as it is meant to be a method of a test class. Given
        that it is set up as a method of a class, it implicitly uses 'self' to access
        the necessary test runner methods for testing the language and vision components.

        Raises:
            This function does not explicitly raise any exceptions, but any exceptions
            raised during the testing of the components would be propagated up to the
            caller of the function.

        """
        from mlx_vlm.models import idefics2

        text_config = idefics2.TextConfig(
            model_type="mistral",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            num_key_value_heads=8,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = idefics2.VisionConfig(
            model_type="idefics2",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=980,
            patch_size=14,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        perceiver_config = idefics2.PerceiverConfig(
            model_type="idefics2Perceiver",
            resampler_n_latents=64,
            resampler_depth=3,
            resampler_n_heads=16,
            resampler_head_dim=96,
            num_key_value_heads=4,
        )

        args = idefics2.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            perceiver_config=perceiver_config,
            model_type="idefics2",
            ignore_index=-100,
            image_token_index=32001,
        )

        model = idefics2.Model(args)

        self.language_test_runner(
            model.language_model,
            args.text_config.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

        self.vision_test_runner(
            model.vision_model,
            args.vision_config.model_type,
            args.vision_config.hidden_size,
            args.vision_config.num_channels,
            (args.vision_config.image_size, args.vision_config.image_size),
        )

    def test_paligemma(self):
        """
        Tests the `paligemma` model by creating instances of its language model, multimodal projector, and vision tower components, and running them through their respective test runners.

        Args:
            self:
                 An instance of a test class that includes `language_test_runner`, `mm_projector_test_runner`, and `vision_test_runner` methods used for conducting
                the necessary tests on the different components of the `paligemma` model.

        Raises:
            Various exceptions can be raised depending if the underlying test runners encounter issues. Exceptions are dependent on the specific implementations
            of the test runner methods.

        """
        from mlx_vlm.models import paligemma

        text_config = paligemma.TextConfig(
            model_type="gemma",
            hidden_size=2048,
            num_hidden_layers=18,
            intermediate_size=16384,
            num_attention_heads=8,
            rms_norm_eps=1e-6,
            vocab_size=257216,
            num_key_value_heads=1,
            rope_theta=10000.0,
            rope_traditional=False,
        )

        vision_config = paligemma.VisionConfig(
            model_type="siglip_vision_model",
            num_hidden_layers=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            projection_dim=2048,
            num_channels=3,
            layer_norm_eps=1e-6,
        )

        args = paligemma.ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="paligemma",
            ignore_index=-100,
            image_token_index=257152,
            hidden_size=2048,
            vocab_size=257216,
        )

        model = paligemma.Model(args)

        self.language_test_runner(
            model.language_model,
            args.text_config.model_type,
            args.text_config.vocab_size,
            args.text_config.num_hidden_layers,
        )

        self.mm_projector_test_runner(
            model.multi_modal_projector,
            args.vision_config.hidden_size,
            args.text_config.hidden_size,
        )

        self.vision_test_runner(
            model.vision_tower,
            args.vision_config.model_type,
            args.vision_config.hidden_size,
            args.vision_config.num_channels,
            (args.vision_config.image_size, args.vision_config.image_size),
        )


if __name__ == "__main__":
    unittest.main()
