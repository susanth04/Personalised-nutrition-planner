import cobra
import traceback

try:
    # Define compartments
    c = "c"  # cytosol
    e = "e"  # extracellular
    
    print("Creating model...")
    model = cobra.Model("simplified_gut_model")
    
    # Define metabolites
    print("Defining metabolites...")
    fiber_e = cobra.Metabolite("fiber_e", compartment=e)
    fiber_c = cobra.Metabolite("fiber_c", compartment=c)
    acetate_c = cobra.Metabolite("acetate_c", compartment=c)
    butyrate_c = cobra.Metabolite("butyrate_c", compartment=c)
    butyrate_e = cobra.Metabolite("butyrate_e", compartment=e)
    
    # Add energy metabolites
    atp_c = cobra.Metabolite("atp_c", compartment=c)
    adp_c = cobra.Metabolite("adp_c", compartment=c)
    pi_c = cobra.Metabolite("pi_c", compartment=c)
    h2o_c = cobra.Metabolite("h2o_c", compartment=c)
    h_c = cobra.Metabolite("h_c", compartment=c)
    nadh_c = cobra.Metabolite("nadh_c", compartment=c)
    nad_c = cobra.Metabolite("nad_c", compartment=c)
    
    # Create transport reactions
    print("Creating reactions...")
    fiber_transport = cobra.Reaction("R_fiber_transport")
    fiber_transport.add_metabolites({fiber_e: -1, fiber_c: 1})
    
    # Exchange reactions
    fiber_uptake = cobra.Reaction("EX_fiber_e")
    fiber_uptake.add_metabolites({fiber_e: -1})
    
    # Metabolic reactions
    fiber_to_acetate = cobra.Reaction("R_fiber_to_acetate")
    fiber_to_acetate.add_metabolites({
        fiber_c: -1,
        acetate_c: 2,
        atp_c: 1,
        adp_c: -1,
        pi_c: -1,
        h2o_c: -1,
        h_c: 1
    })
    
    acetate_to_butyrate = cobra.Reaction("R_acetate_to_butyrate")
    acetate_to_butyrate.add_metabolites({
        acetate_c: -2,
        butyrate_c: 1,
        h2o_c: -1,
        nadh_c: -1,
        nad_c: 1,
        h_c: 2
    })
    
    butyrate_export = cobra.Reaction("R_butyrate_export")
    butyrate_export.add_metabolites({butyrate_c: -1, butyrate_e: 1})
    
    butyrate_exchange = cobra.Reaction("EX_butyrate_e")
    butyrate_exchange.add_metabolites({butyrate_e: -1})
    
    # Add all reactions to model
    print("Adding reactions to model...")
    model.add_reactions([
        fiber_transport, fiber_uptake, fiber_to_acetate,
        acetate_to_butyrate, butyrate_export, butyrate_exchange
    ])
    
    # Set reaction bounds
    print("Setting reaction bounds...")
    for reaction in model.reactions:
        reaction.bounds = (-1000, 1000)
    
    # Set objective
    print("Setting objective...")
    model.objective = "EX_butyrate_e"
    model.objective_direction = "max"
    
    # Run FBA
    print("Running FBA...")
    solution = model.optimize()
    
    print(f"Solution status: {solution.status}")
    if solution.status == 'optimal':
        print("Fluxes:")
        for reaction_id, flux in solution.fluxes.items():
            if abs(flux) > 1e-6:  # Only print non-zero fluxes
                print(f"{reaction_id}: {flux}")
    else:
        print(f"No optimal solution found: {solution.status}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc() 