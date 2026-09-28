import copernicusmarine

result = copernicusmarine.describe(contains=["cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D"])

for product in result.products:
    for dataset in product.datasets:
        if dataset.dataset_id == "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D":
            for version in dataset.versions:
                print(f"Version label: {version.label}")
                for part in version.parts:
                    for service in part.services:
                        print(f"  Service: {service.service_name}")
                        if service.variables:
                            for var in service.variables:
                                for coord in var.coordinates:
                                    if coord.coordinate_id == "time":
                                        print(f"    Time range: {coord.minimum_value} to {coord.maximum_value}")