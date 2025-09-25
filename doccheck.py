import inspect as I
import proyecto_ml.pipelines.dataprep.nodes as n
import proyecto_ml.pipelines.dataprep.pipeline as p
import proyecto_ml.pipeline_registry as r

objs = [
    n.basic_clean,
    n.iqr_clip,
    n.engineer_features,
    n.prep,
    p.create_pipeline,
    r.register_pipelines,
]

for o in objs:
    doc = I.getdoc(o) or "(sin docstring)"
    print(f"\n>>> {o.__module__}.{o.__name__}\n{doc[:300]} ...")
