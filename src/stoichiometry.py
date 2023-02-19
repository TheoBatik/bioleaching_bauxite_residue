# Stoichiometric coefficients of reactants and products
reactants = [
    # Citric acid consumption
    {'ScO(OH)': 1, 'C6H8O7': 3}, # Scandium goethite
    {'Y2O3': 1, 'C6H8O7': 6}, # Yttrium oxide
    {'Fe2O3': 1, 'C6H8O7': 6}, 
    {'Al2O3': 1, 'C6H8O7': 3},
    {'CaO': 1, 'C6H8O7': 1},
    {'TiO2': 1, 'C6H8O7': 4},
    # Oxalic acid consumption
    {'ScO(OH)': 2, 'C2H2O4': 3},
    {'Fe2O3': 1, 'C2H2O4': 3},
    {'Al2O3': 1, 'C2H2O4': 3},
    {'CaO': 1, 'C2H2O4': 1},
    {'TiO2': 1, 'C2H2O4': 2},
    # # Gluconic acid consumption
    # {'Fe2O3': 1, 'C6H12O7': 6},
    # # Iron(II) Oxalate formation
    # {'Fe': 1, '(C2O4)2-': 1},
    # Sc Oxalate formation
    # {'Sc3+': 2, '(C2O4)2-': 3},
    # # Acid production
    # {'C12H22O11': 1}
]
products = [
    # Citric acid consumption
    {'Sc3+': 1, 'C6H7O7-': 3 }, # Scandium
    {'Y3+': 2, 'C2H2O4-': 6 }, # Yttrium
    {'Fe3+': 2, 'C6H7O7-': 6 }, # Iron
    {'Al3+': 2, 'C2H2O4-': 3}, # Aluminium
    {'Ca2+': 1, 'C2H2O4-': 1}, #, 'H+': 3 },
    {'Ti4+': 1, 'C2H2O4-': 4}, #, 'H+': 3},
    # Oxalic acid consumption
    {'Sc3+': 2, '(C2O4)2-': 3 }, #, 'H+': 2 }, #, 'H2O': 2},
    {'Fe3+': 2, '(C2O4)2-': 3 }, #, 'H+': 6 }, #, 'H2O': 3},
    {'Al3+': 2, '(C2O4)2-': 3 }, #, 'H+': 6 },
    {'Ca2+': 1, '(C2O4)2-': 1 }, #, 'H+': 2 },
    {'Ti4+': 1, '(C2O4)2-': 2 }, #, 'H+': 2},
    # Gluconic acid consumption
    # {'Fe3+': 2, 'C6H11O7': 6},
    # # Iron(II) Oxalate formation
    # {'FeC2O4'},
    # Sc Oxalate formation
    # {'Sc2(C2O4)'},
    # # Acid production
    # {'C6H8O7': 1}
]








# Stoichiometric coefficients of reactants and products
reactants_2 = [
    # Acid dissolution reactions
    {'C6H8O7': 1},
    # {'H+': 1, 'C6H7O7-': 1},
    # {'C2H2O4-': 1},
    # {'H+': 2, '(C2O4)2-': 1},
    # {'C2H2O4-': 1},
    # Citric acid consumption
    # {'ScO(OH)': 1, 'H+': 3}, # Scandium Oxide
    # {'Y2O3': 1, 'C6H8O7': 6}, # Yttrium Oxide
    {'Fe2O3': 1, 'H+': 6}, # Iron Oxide
    # {'Al2O3': 1, 'C6H8O7': 3},
    # {'CaO': 1, 'C6H8O7': 1},
    # {'TiO2': 1, 'C6H8O7': 2},
    # Oxalic acid consumption
    # {'ScO(OH)': 1, 'C2H2O4-': 1},
    # {'Fe2O3': 1, 'C2H2O4-': 3},
    # {'Al2O3': 1, 'C2H2O4-': 3},
    # {'CaO': 1, 'C2H2O4-': 1},
    # {'TiO2': 1, 'C2H2O4-': 2},
    # Gluconic acid consumption
    # {'Fe2O3': 1, 'C6H12O7': 6},
    # Sc Oxalate formation
    # {'Sc3+': 1, '(C2O4)2-': 1},
    # Acid production
    # {'C12H22O11': 1}
]
products_2 = [
    # Acid dissolution reactions
    {'H+': 1, 'C6H7O7-': 1},
    # {'C6H8O7': 1},
    # {'H+': 2, '(C2O4)2-': 1},
    # {'C2H2O4-': 1},
    # Citric acid consumption
    # {'Sc3+': 1 }, #, 'C6H7O7-': 3 }, #,'H2O': 2}, # }, 
    # {'Y': 2 }, #, 'C6H7O7-': 6 }, # 'H2O': 3}, # Yttrium
    {'Fe3+': 2 }, #, 'C6H7O7-': 6 }, # 'H2O': 3}, #
    # {'Al3+': 2, 'C6H7O7-': 3}, #, 'H+': 3 },
    # {'Ca2+': 1, 'C6H7O7-': 1}, #, 'H+': 3 },
    # {'Ti4+': 1, 'C6H7O7-': 1}, #, 'H+': 3},
    # Oxalic acid consumption
    # {'Sc3+': 1, '(C2O4)2-': 1 }, #, 'H+': 2 }, #, 'H2O': 2},
    # {'Fe3+': 2, 'C204': 3 }, #, 'H+': 6 }, #, 'H2O': 3},
    # {'Al3+': 2, '(C2O4)2-': 3 }, #, 'H+': 6 },
    # {'Ca2+': 1, '(C2O4)2-': 1 } #, 'H+': 2 },
    # {'Ti4+': 1, '(C2O4)2-': 1 } #, 'H+': 2},
    # Gluconic acid consumption
    # {'Fe3+': 2, 'C6H11O7': 6, 'H+': 2},
    # Sc Oxalate formation
    # {'ScC2O4'},
    # Acid production
    # {'C6H8O7': 1}
]