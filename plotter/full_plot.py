import matplotlib.pyplot as plt
from plotter.plotter import set_default, main

# Varying sparsity (fix sparsity + nmu)
set_default("rigl",nmu= 1000, pr=0.9, variable = "sparsity")
main()
set_default("set",nmu= 1000,pr=0.9, variable ="sparsity")
main()
set_default("rigl",nmu=1000, pr=0.3, variable ="sparsity")
main()
set_default("set",nmu=1000, pr=0.3, variable ="sparsity")
main()

# Varying pruning ratio (fix sparsity + nmu)
set_default("rigl",nmu= 1000, s=0.7, variable ="pruning_ratio")
main()
set_default("set",nmu=1000, s=0.7, variable ="pruning_ratio")
main()
set_default("rigl",nmu=1000, s=0.5, variable ="pruning_ratio")
main()
set_default("set",nmu=1000, s=0.7, variable ="pruning_ratio")
main()

# Varying num_mask_updates (fix sparsity + pruning_ratio)
set_default("rigl",s= 0.7, pr=0.7, variable ="num_mask_updates")
main()
set_default("set",s=0.1, pr=0.7, variable ="num_mask_updates")
main()
set_default("rigl",s= 0.5, pr=0.9, variable ="num_mask_updates")
main()
set_default("set",s=0.5, pr=0.3, variable ="num_mask_updates")
main()

