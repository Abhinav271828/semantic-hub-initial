from analysis.trends import plot_evolution_trends
from torch import tensor

run_name = "5iipl67m"
results = eval(open(f"{run_name}_trends.txt", "r").read())
# {accuracy_trends: Dict[int, float],
#  distance_bw_datatypes: Dict[int, float],
#  intervention_trends: {'empty': Dict[int, float],
#                        'invalid': Dict[int, float],
#                        'valid': Dict[int, float]},
#  overall_distances: Dict[int, float],
#  generalization_trends: Dict[int, float]
# in_distribution_trends: Dict[int, float]
# }

ceiling = 3_001


def break_dict_at(d):
    return {k: v for k, v in d.items() if int(k) <= ceiling}


plot_evolution_trends(
    distance_bw_datatypes=break_dict_at(results["distance_bw_datatypes"]),
    overall_avg_distances=break_dict_at(results["overall_avg_distances"]),
    accuracy_trends=break_dict_at(results["accuracy_trends"]),
    intervention_trends={
        k: break_dict_at(v) for k, v in results["intervention_trends"].items()
    },
    generalization_trends=results["generalization_trends"],
    in_distribution_trends=results["in_distribution_trends"],
    save_path=f"{run_name}_{ceiling}",
)
