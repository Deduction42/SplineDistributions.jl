import Pkg
Pkg.add("PkgTemplates")

using PkgTemplates
tpl = Template(;
    plugins = [!Git, !License, !TagBot, !GitHubActions]
)

tpl("WearRateDistributions")