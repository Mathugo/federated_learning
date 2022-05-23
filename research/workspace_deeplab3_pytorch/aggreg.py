from openfl.component.aggregation_functions import AggregationFunction, WeightedAverage
from openfl.component.aggregation_functions.core import AdaptiveAggregation
from openfl.utilities.optimizers.numpy.base_optimizer import Optimizer
import numpy as np

""" optimizer for the aggregation
You can define your own numpy based optimizer, which will be used for global model 
aggreagation, e.i -> 

my_own_optimizer = MyOpt(model_interface=MI, learning_rate=0.01)
agg_fn = AdaptiveAggregation(optimizer=my_own_optimizer,
                             agg_func=WeightedAverage()"""

class MyOpt(Optimizer):
    """My optimizer implementation."""

    def __init__(
        self,
        *,
        params: Optional[Dict[str, np.ndarray]] = None,
        model_interface=None,
        learning_rate: float = 0.001,
        param1: Any = None,
        param2: Any = None
    ) -> None:
        """Initialize.

        Args:
            params: Parameters to be stored for optimization.
            model_interface: Model interface instance to provide parameters.
            learning_rate: Tuning parameter that determines
                the step size at each iteration.
            param1: My own defined parameter.
            param2: My own defined parameter.
        """
        super().__init__()
        pass # Your code here!

    def step(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Perform a single step for parameter update.

        Implement your own optimizer weights update rule.

        Args:
            gradients: Partial derivatives with respect to optimized parameters.
        """
        pass # Your code here!

""" This is an example of a custom tensor clipping aggregation function 
that multiplies all local tensors by 0.3 and averages them according 
to weights equal to data parts to produce the resulting global tensor."""

class ClippedAveraging(AggregationFunction):
    def __init__(self, ratio):
        self.ratio = ratio

    def call(self,
            local_tensors,
            db_iterator,
            tensor_name,
            fl_round,
            *__):
        """Aggregate tensors.

        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training result.
                        These tensors are passed to the aggregator node after local learning.
                    - 'aggregated' indicates that tensor is a result of aggregation.
                        These tensors are sent to collaborators for the next round.
                    - 'delta' indicates that value is a difference between rounds
                        for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            np.ndarray: aggregated tensor
        """
        clipped_tensors = []
        previous_tensor_value = None
        for record in db_iterator:
            if (
                record['round'] == (fl_round - 1)
                and record['tensor_name'] == tensor_name
                and 'aggregated' in record['tags']
                and 'delta' not in record['tags']
            ):
                previous_tensor_value = record['nparray']
        weights = []
        for local_tensor in local_tensors:
            prev_tensor = previous_tensor_value if previous_tensor_value is not None else local_tensor.tensor
            delta = local_tensor.tensor - prev_tensor
            new_tensor = prev_tensor + delta * self.ratio
            clipped_tensors.append(new_tensor)
            weights.append(local_tensor.weight)

        return np.average(clipped_tensors, weights=weights, axis=0)