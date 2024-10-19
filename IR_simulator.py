import streamlit as st


from openfisca_core import reforms, periods
from openfisca_france.model.base import *
from openfisca_france.scenarios import init_single_entity

from openfisca_france import FranceTaxBenefitSystem

from openfisca_core.simulation_builder import SimulationBuilder

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio

IR_one_shot = None


# Modification bareme 2024 (projet de loi de finances 2025)
def modifier_un_parametre(parameters):
    # Ceci décrit la periode sur laquelle va s'appliquer ce changement
    reform_year = 2024
    reform_period = periods.period(reform_year)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[1].threshold.update(start=reform_period.start, value=11520)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[2].threshold.update(start=reform_period.start, value=29373)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[3].threshold.update(start=reform_period.start, value=83988)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[4].threshold.update(start=reform_period.start, value=180648)
    return parameters

# Cette partie rassemble les changements de paramètre dans une seule réforme appelée ici MaReforme
class MaReforme(reforms.Reform):
    def apply(self):
        self.modify_parameters(modifier_function = modifier_un_parametre)


simulation_builder = SimulationBuilder()
#if 'initialized' not in st.session_state or not st.session_state.initialized:         
legislation_france = FranceTaxBenefitSystem()
legislation_reforme = MaReforme(legislation_france)
#    st.session_state.initialized = True

with st.sidebar:
    annee_simulation = st.selectbox("Annee de perception des revenus", (2024, 2023, 2022))

    nb_adultes = st.selectbox("Nombre d'adultes", (1, 2))
    if nb_adultes == 1:
        parent_isole = st.checkbox('Parent isolé?')
    nb_enfants = st.selectbox("Nombre d'enfants à charge", (0, 1, 2, 3, 4, 5))
    nb_alternes = st.selectbox("Nombre d'enfants en garde alternée", (0, 1, 2, 3, 4, 5))
    

individus = {}
for i in range(nb_adultes):
    individus['parent' + str(i+1)] = {}
for i in range(nb_enfants):
    individus['enfant' + str(i+1)] = {'enfant_a_charge': {annee_simulation: True}}
for i in range(nb_alternes):
    individus['enfant' + str(i+1 + nb_enfants)] = {'garde_alternee': {annee_simulation: True}}


#individus['parent1']['salaire_imposable']= {annee_simulation : salary}
    
foyer_fiscal = {'foyerfiscal1': {'declarants': ['parent' + str(i+1) for i in range(nb_adultes)],
                                 #'personnes_a_charge': ['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)],
                                 'nbH': {annee_simulation: nb_alternes},
                                 'nb_pac':{annee_simulation: nb_enfants}
                                 }}

if nb_adultes == 1:
    if parent_isole:
        foyer_fiscal['foyerfiscal1']['caseT'] =  {annee_simulation: True}
    

CASE = {}
#CASE['familles'] = {'individus': individus}
CASE['individus'] = individus
CASE['foyers_fiscaux'] = foyer_fiscal


#st.write(individus)
#st.write(foyer_fiscal)
#st.write(CASE)

def calcul_IR_oneshot(CASE):
    simulation_IR = simulation_builder.build_from_entities(legislation_reforme, CASE)
    IR_one_shot = simulation_IR.calculate('ir_plaf_qf', str(annee_simulation))[0]
    Revenu_one_shot = simulation_IR.calculate('rbg', str(annee_simulation))[0]
    
    return IR_one_shot, Revenu_one_shot

tab_one_shot, tab_evolution, tab_PER = st.tabs(["Calcul de l'IR", "Evolution de l'IR", "Versement Optimal PER"])

with tab_one_shot:
    col1, col2 = st.columns(2, vertical_alignment='bottom')
    salary = col1.number_input("Salaire", 40000)
    #st.divider()
    button_run = col2.button("Calcul", icon="😰", use_container_width=True)



    if(button_run):
        CASE['individus']['parent1']['salaire_imposable']= {annee_simulation : salary}

        simulation_IR = simulation_builder.build_from_entities(legislation_reforme, CASE)
        IR_one_shot = simulation_IR.calculate('ir_plaf_qf', str(annee_simulation))[0]
        Revenu_one_shot = simulation_IR.calculate('rbg', str(annee_simulation))[0]
        nb_part_one_shot = simulation_IR.calculate('nbptr', str(annee_simulation))[0]

        #CASE['individus']['parent1']['salaire_imposable']= {annee_simulation : salary}
        #IR_one_shot, Revenu_one_shot = calcul_IR_oneshot(CASE)
        
        st.divider()
        
        def format_space_thousand_sep(num, trailing=" €"):
            return '{:,}'.format(np.round(num, 0)).replace(',', ' ') + trailing
        
        col_one_shot_1, col_one_shot_2, col_one_shot_3 = st.columns(3)
        col_one_shot_1.metric("Revenu imposable", format_space_thousand_sep(Revenu_one_shot), "")
        col_one_shot_2.metric("Nombre de parts", str(nb_part_one_shot), "")
        col_one_shot_3.metric("Impot sur le revenu",  format_space_thousand_sep(IR_one_shot), "")
        
        IR_ss_qf_one_shot = simulation_IR.calculate('ir_ss_qf', str(annee_simulation))[0]
        avantage_qf_one_shot = -simulation_IR.calculate('avantage_qf', str(annee_simulation))[0]
        
        TMI_one_shot = simulation_IR.calculate('ir_taux_marginal', str(annee_simulation))[0]
        
        taux_moyen_imposition_one_shot = simulation_IR.calculate('taux_moyen_imposition', str(annee_simulation))[0]
        col_one_shot_1.metric("Effet quotient familial", 
                              format_space_thousand_sep(avantage_qf_one_shot), 
                              str(np.round(avantage_qf_one_shot/IR_ss_qf_one_shot*100, 1)) + " %", delta_color = 'inverse')
        col_one_shot_2.metric("Tranche marginale d'imposition",
                              str(np.round(TMI_one_shot * 100, 0)) + "%", "")
        col_one_shot_3.metric("Taux moyen d'imposition",
                              str(np.round(taux_moyen_imposition_one_shot * 100, 1)) + "%", "")
        
    #   st.write(simulation_IR.calculate('rni', str(annee_simulation)))
    #   st.write(simulation_IR.calculate('rng', str(annee_simulation)))
with tab_evolution:
    CASE_EVOL = CASE.copy()
    CASE_EVOL['axes']= [[{'count':100, 'name':'salaire_imposable', 'min':0, 'max':100000, 'period': annee_simulation}]]
    
    CASE_EVOL['familles'] = {'famille1': {}}
    CASE_EVOL['familles']['famille1'] = {'parents': ['parent' + str(i+1) for i in range(nb_adultes)],
                                            'enfants': ['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]}
    CASE_EVOL['menages'] = {'menage1': {}}
    CASE_EVOL['menages']['menage1'] = {'personne_de_reference': ['parent1'],
                                        'enfants':['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]}
    if nb_adultes > 1:
        CASE_EVOL['menages']['menage1']['conjoint'] = ['parent2']    
    
    n_reshape = nb_adultes + nb_alternes + nb_enfants
    simulation_builder = SimulationBuilder()
    simulation_evol = simulation_builder.build_from_entities(legislation_reforme, CASE_EVOL)
    sdb = simulation_evol.calculate_add('salaire_imposable', annee_simulation)
    salaire_foyer = sdb.reshape(100,n_reshape).sum(1)

    income_tax_evol = simulation_evol.calculate('ir_plaf_qf', annee_simulation)

    income_tax_evol = income_tax_evol.reshape(100,n_reshape - max(0, nb_adultes - 1)).sum(1) 

    df_income_tax = pd.DataFrame.from_dict({'revenu': salaire_foyer, 'IR': income_tax_evol})
    white_bg = st.checkbox('Arrière-plan blanc')
    if white_bg:
        pio.templates.default = "none"
    else:
        pio.templates.default = 'streamlit'
    fig = px.line(df_income_tax, x='revenu', y=['IR'])

    if IR_one_shot is not None:
        fig.add_scatter(x=[salary],
                y=[IR_one_shot],
                marker=dict(
                    color='red',
                    size=10
                ),
                name='IR actuel')
    
    fig.update_layout(
        title="Evolution Impôt sut le revenu",
        xaxis_title="Revenu net annuel",
        yaxis_title="Montant de l'IR",
        xaxis_ticksuffix = '€',
        yaxis_ticksuffix = '€')
    st.plotly_chart(fig)

with tab_PER:
    ir_tranche = simulation_IR.calculate('ir_tranche', annee_simulation)
    
    CASE_PER = CASE.copy()
    CASE_PER['axes']= [[{'count':int(salary), 'name':'f6rs', 'min':0, 'max':salary, 'period': annee_simulation}]]
    
    CASE_PER['familles'] = {'famille1': {}}
    CASE_PER['familles']['famille1'] = {'parents': ['parent' + str(i+1) for i in range(nb_adultes)],
                                            'enfants': ['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]}
    CASE_PER['menages'] = {'menage1': {}}
    CASE_PER['menages']['menage1'] = {'personne_de_reference': ['parent1'],
                                        'enfants':['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]}
    if nb_adultes > 1:
        CASE_PER['menages']['menage1']['conjoint'] = ['parent2']    
    n_reshape = nb_adultes + nb_alternes + nb_enfants
    simulation_builder = SimulationBuilder()
    simulation_PER = simulation_builder.build_from_entities(legislation_reforme, CASE_PER)
    versement_PER = simulation_PER.calculate_add('f6rs', annee_simulation)
    
    income_tax_PER = simulation_PER.calculate('ir_plaf_qf', annee_simulation)
    tmi_PER = simulation_PER.calculate('ir_tranche', annee_simulation)
    
    df_income_tax_PER = pd.DataFrame.from_dict({'Versement_PER': versement_PER, 'IR': income_tax_PER})
    fig_PER = px.line(df_income_tax_PER, x='Versement_PER', y=['IR'])
    st.plotly_chart(fig_PER)
    df_tmi = pd.DataFrame.from_dict({'Versement_PER':versement_PER, 'TMI': tmi_PER})
    tranches_bareme_en_cours = legislation_reforme.parameters.impot_revenu.bareme_ir_depuis_1945.bareme(annee_simulation).rates
    df_tmi['TMI'] = df_tmi['TMI'].apply(lambda x: tranches_bareme_en_cours[x])

    fig_tmi_per = px.line(df_tmi, x='Versement_PER', y='TMI')
    versement_optimal_PER = df_tmi[df_tmi['TMI'] == .11].head(1)['Versement_PER'].values[0]
    economie_impot_PER = df_income_tax_PER[df_income_tax_PER['Versement_PER'] == versement_optimal_PER]
    st.dataframe(df_income_tax_PER, width=600)
    st.dataframe(economie_impot_PER)
    
    col1_PER, col2_PER, col3_PER = st.columns(3)
    col1_PER.metric("Versement Optimal PER", format_space_thousand_sep(versement_optimal_PER))
    col2_PER.metric("Economie d'impôts", format_space_thousand_sep(economie_impot_PER), "")
    
    
    st.plotly_chart(fig_tmi_per)



    
    
    
    
    



    
    
    
