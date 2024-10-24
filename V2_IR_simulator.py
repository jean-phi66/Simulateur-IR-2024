import streamlit as st
from streamlit import session_state as ss

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

step = 100
round_num = -2

def round_to(num, round_num=round_num):
    return int(np.round(num, round_num))

# Modification bareme 2024 (projet de loi de finances 2025)
# https://www.boursier.com/patrimoine/impots/actualites/impot-sur-le-revenu-les-nouveaux-baremes-2025-pour-le-calcul-de-la-decote-et-des-avantages-familiaux-8983.html#:~:text=La%20décote%20consiste%20à%20réduire,les%20célibataires%2C%20divorcés%20ou%20veufs.
def modifier_un_parametre(parameters):
    # Ceci décrit la periode sur laquelle va s'appliquer ce changement
    reform_year = 2024
    reform_period = periods.period(reform_year)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[1].threshold.update(start=reform_period.start, value=11520)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[2].threshold.update(start=reform_period.start, value=29373)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[3].threshold.update(start=reform_period.start, value=83988)
    parameters.impot_revenu.bareme_ir_depuis_1945.bareme[4].threshold.update(start=reform_period.start, value=180648)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.decote.seuil_celib.update(start=reform_period, value=890.)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.decote.seuil_couple.update(start=reform_period, value=1473.)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.plafond_avantages_procures_par_demi_part.general.update(start=reform_period, value=1794.)
    parameters.impot_revenu.calcul_impot_revenu.plaf_qf.plafond_avantages_procures_par_demi_part.celib_enf.update(start=reform_period, value=4232)
    
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
    annee_simulation = st.selectbox("Annee de perception des revenus", (2024, 2023, 2022, 2021, 2020))

    nb_adultes = st.selectbox("Nombre d'adultes", (1, 2))
    if nb_adultes == 1:
        parent_isole = st.checkbox('Parent isolé?')
    nb_enfants = st.selectbox("Nombre d'enfants à charge", (0, 1, 2, 3, 4, 5))
    nb_alternes = st.selectbox("Nombre d'enfants en garde alternée", (0, 1, 2, 3, 4, 5))
    col1, col2 = st.columns(2, vertical_alignment='bottom')
    salary = col1.number_input("Salaire", value=50000, step=step)
    button_run = col2.button("Calcul", icon="😰", use_container_width=True)
    plafond_PER = st.number_input("Plafond PER", value=16000)
    
    ss['salary'] = salary

    

if button_run:
    # Creation des entitées utilisées par openfisca
    individus = {}
    for i in range(nb_adultes):
        individus['parent' + str(i+1)] = {}
    for i in range(nb_enfants):
        individus['enfant' + str(i+1)] = {'enfant_a_charge': {annee_simulation: True}}
    for i in range(nb_alternes):
        individus['enfant' + str(i+1 + nb_enfants)] = {'garde_alternee': {annee_simulation: True}}
        
    foyer_fiscal = {'foyerfiscal1': {'declarants': ['parent' + str(i+1) for i in range(nb_adultes)],
                                    #'personnes_a_charge': ['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)],
                                    'nbH': {annee_simulation: nb_alternes},
                                    'nb_pac':{annee_simulation: nb_enfants}
                                    }}

    if nb_adultes == 1:
        if parent_isole:
            foyer_fiscal['foyerfiscal1']['caseT'] =  {annee_simulation: True}
        

    CASE = {}
    CASE['individus'] = individus
    CASE['foyers_fiscaux'] = foyer_fiscal
    
    #length_simu = int(np.round(salary * 1.2, round_num))
    length_simu = round_to(salary * 1.2)
    length_simu_n = int(length_simu/step)
    
    CASE['axes']= [[{'count':length_simu_n, 'name':'rbg', 'min':0, 'max':length_simu,
                     'period': annee_simulation}]]
        
    CASE['familles'] = {'famille1': {}}
    CASE['familles']['famille1'] = {'parents': ['parent' + str(i+1) for i in range(nb_adultes)],
                                            'enfants': ['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]}
    CASE['menages'] = {'menage1': {}}
    CASE['menages']['menage1'] = {'personne_de_reference': ['parent1'],
                                        'enfants':['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]}
    if nb_adultes > 1:
        CASE['menages']['menage1']['conjoint'] = ['parent2']    
    
    # Simulation IR
    n_reshape = nb_adultes + nb_alternes + nb_enfants
    simulation_builder = SimulationBuilder()
    simulation_evol = simulation_builder.build_from_entities(legislation_reforme, CASE)
    
    nb_part_one_shot = simulation_evol.calculate('nbptr', str(annee_simulation))[0]

    sdb = simulation_evol.calculate_add('rbg', annee_simulation)
    salaire_foyer = sdb.reshape(int(length_simu_n),n_reshape- max(0, nb_adultes - 1)).sum(1)
    
    income_tax_evol = simulation_evol.calculate('ip_net', annee_simulation)
    income_tax_evol = income_tax_evol.reshape(int(length_simu_n),n_reshape - max(0, nb_adultes - 1)).sum(1) 
    
    avantage_qf_evol = -simulation_evol.calculate('avantage_qf', str(annee_simulation))
    avantage_qf_evol = avantage_qf_evol.reshape(int(length_simu_n),n_reshape- max(0, nb_adultes - 1)).sum(1)
    
    TMI_evol = simulation_evol.calculate('ir_taux_marginal', str(annee_simulation))
    TMI_evol = TMI_evol.reshape(int(length_simu_n),n_reshape- max(0, nb_adultes - 1)).sum(1)
    
    taux_moyen_imposition_evol = simulation_evol.calculate('taux_moyen_imposition', str(annee_simulation))
    taux_moyen_imposition_evol = taux_moyen_imposition_evol.reshape(int(length_simu_n),n_reshape- max(0, nb_adultes - 1)).sum(1)
    
    IR_ss_qf_evol = simulation_evol.calculate('ir_ss_qf', str(annee_simulation))
    IR_ss_qf_evol = IR_ss_qf_evol.reshape(int(length_simu_n),n_reshape - max(0, nb_adultes - 1)).sum(1)
    
    decote_evol = simulation_evol.calculate('decote_gain_fiscal', str(annee_simulation))
    decote_evol = decote_evol.reshape(int(length_simu_n),n_reshape - max(0, nb_adultes - 1)).sum(1)
    
    df_income_tax_evol = pd.DataFrame.from_dict({'Revenu': salaire_foyer, 'IR': income_tax_evol,
                                                 'TMI': TMI_evol,
                                                 'Taux moyen d imposition' :taux_moyen_imposition_evol,
                                                 'IR sans QF' : IR_ss_qf_evol,
                                                 'Reduction QF': avantage_qf_evol,
                                                 'Decote': decote_evol})
    
    df_income_tax_evol['Revenu'] = df_income_tax_evol['Revenu'].apply(round_to)#.apply(int)
    ss['df_income_tax_evol'] = df_income_tax_evol

    df_one_shot = df_income_tax_evol[df_income_tax_evol['Revenu'] == int(np.round(salary * .9, round_num))]
    
    # Montant optimal PER
    CASE_PER = CASE.copy()
    CASE_PER['foyers_fiscaux']['foyerfiscal1']['rbg'] = {annee_simulation: salary * .9}
    length_simu_PER = int(salary)
    CASE_PER['axes']= [[{'count':length_simu_PER, 'name':'f6rs', 'min':0, 'max':length_simu_PER, 'period': annee_simulation}]]
    
    n_reshape = nb_adultes + nb_alternes + nb_enfants
    
    simulation_builder = SimulationBuilder()
    simulation_PER = simulation_builder.build_from_entities(legislation_reforme, CASE_PER)
    
    versement_PER = simulation_PER.calculate_add('f6rs', annee_simulation)
    versement_PER = versement_PER.reshape(int(length_simu_PER),n_reshape).sum(1)
    
    income_tax_PER = simulation_PER.calculate('ip_net', annee_simulation)
    income_tax_PER = income_tax_PER.reshape(int(length_simu_PER),n_reshape- max(0, nb_adultes - 1)).min(1)
    
    TMI_PER = simulation_PER.calculate('ir_taux_marginal', annee_simulation)
    TMI_PER = TMI_PER.reshape(int(length_simu_PER),n_reshape- max(0, nb_adultes - 1))
    TMI_PER = TMI_PER.min(1)
    
    df_income_tax_PER = pd.DataFrame.from_dict({'Versement_PER': versement_PER, 
                                                'IR': income_tax_PER,
                                                'TMI': TMI_PER})
    nb_part_PER = simulation_PER.calculate('nbptr', str(annee_simulation))[0]
    
    df_optimal_PER = df_income_tax_PER[df_income_tax_PER['TMI'] == .11].head(1)
    
    ss['df_income_tax_PER'] = df_income_tax_PER
    
    if len(df_optimal_PER >0):
        versement_optimal_PER = df_optimal_PER['Versement_PER'].values[0]
        impot_PER = df_optimal_PER['IR'].values[0]
    else:
        versement_optimal_PER = 0.
        impot_PER = 0.
    
    versement_PER_TMI = versement_optimal_PER
    
    if(versement_optimal_PER > plafond_PER):
        df_optimal_PER = df_income_tax_PER[df_income_tax_PER['Versement_PER'] == plafond_PER].head(1)
        
        versement_optimal_PER = df_optimal_PER['Versement_PER'].values[0]
        impot_PER = df_optimal_PER['IR'].values[0]
        

    def format_space_thousand_sep(num, trailing=" €"):
        return '{:,}'.format(np.round(num, 0)).replace(',', ' ') + trailing

tab_one_shot, tab_PER_effet, tab_PER_optim = st.tabs(["Calcul de l'IR", "Effet Versement PER","Versement Optimal PER"])
with tab_one_shot:
    if(button_run):
        Revenu_one_shot = df_one_shot['Revenu'].values[0]
        IR_one_shot = df_one_shot['IR'].values[0]
        avantage_qf_one_shot = df_one_shot['Reduction QF'].values[0]
        IR_ss_qf_one_shot = df_one_shot['IR sans QF'].values[0]
        TMI_one_shot = df_one_shot['TMI'].values[0]
        taux_moyen_imposition_one_shot = df_one_shot['Taux moyen d imposition'].values[0]
        decote_one_shot = df_one_shot['Decote'].values[0]
        
        col_one_shot_1, col_one_shot_2, col_one_shot_3 = st.columns(3)
        col_one_shot_1.metric("Revenu imposable", format_space_thousand_sep(Revenu_one_shot), "")
        col_one_shot_2.metric("Nombre de parts", str(nb_part_one_shot), "")
        col_one_shot_3.metric("Impot sur le revenu",  format_space_thousand_sep(IR_one_shot), "")
        
        col_one_shot_1.metric("Effet quotient familial", 
                              format_space_thousand_sep(avantage_qf_one_shot), 
                              str(np.round(avantage_qf_one_shot/IR_ss_qf_one_shot*100, 1)) + " %", delta_color = 'inverse')
        col_one_shot_2.metric("Tranche marginale d'imposition",
                               str(np.round(TMI_one_shot * 100, 0)) + "%", "")
        col_one_shot_3.metric("Taux moyen d'imposition",
                               str(np.round(taux_moyen_imposition_one_shot * 100, 1)) + "%", "")
        
        col_one_shot_1.metric("Decote", 
                        format_space_thousand_sep(-decote_one_shot), 
                        str(np.round(-decote_one_shot/IR_ss_qf_one_shot*100, 1)) + " %", delta_color = 'inverse')

        
        fig = px.line(df_income_tax_evol, x='Revenu', y=['IR'])

        if IR_one_shot is not None:
            fig.add_scatter(x=[salary * .9],
                    y=[IR_one_shot],
                    text=format_space_thousand_sep(IR_one_shot),
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
        
with tab_PER_optim:
    if(button_run):
        
        versement_PER_preco = min(versement_optimal_PER, plafond_PER)
        
        col1_PER, col2_PER, col3_PER = st.columns(3)
        col1_PER.metric("Versement Optimal PER", format_space_thousand_sep(versement_PER_preco))
        col2_PER.metric("Impot avec versement", format_space_thousand_sep(impot_PER), "")
        col3_PER.metric("Economie d'impots", format_space_thousand_sep(IR_one_shot - impot_PER), "")
        col1_PER.metric("Versement Optimal TMI", format_space_thousand_sep(versement_PER_TMI))
        
        fig_PER_evol = px.line(df_income_tax_evol, x='Revenu', y=['IR'])

        if IR_one_shot is not None:
            fig_PER_evol.add_scatter(x=[salary * .9],
                    y=[IR_one_shot],
                    text=format_space_thousand_sep(IR_one_shot),
                    marker=dict(
                        color='red',
                        size=10
                    ),
                    name='IR actuel')
        fig_PER_evol.add_scatter(x=[salary * .9 - versement_PER_preco],
                                 y=[impot_PER],
                                 text=format_space_thousand_sep(IR_one_shot),
                                 marker=dict(
                                     color='green',
                                     size=10
                                     ),
                                 name='IR avec versement')
        st.plotly_chart(fig_PER_evol)
        
        fig_PER = px.line(df_income_tax_PER, x='Versement_PER', y=['IR'])
        st.plotly_chart(fig_PER)

        fig_tmi_PER = px.line(df_income_tax_PER, x='Versement_PER', y=['TMI'])
        st.plotly_chart(fig_tmi_PER)
        
@st.fragment()
def versement_PER_effect_fragment():
    versement_PER_input = st.slider("Versement PER", min_value=0, max_value=int(salary * 0.9), value=0, step=100)

    df_income_tax_PER = ss['df_income_tax_PER']
    df_income_tax_evol = ss['df_income_tax_evol']


    df_versement_PER = df_income_tax_PER[df_income_tax_PER['Versement_PER'] == versement_PER_input].head(1)
    IR_versement_PER = df_versement_PER['IR'].values[0]
    
    col1_vers_PER, col2_vers_PER, col3_vers_PER = st.columns(3)
    col1_vers_PER.metric("Versement PER", format_space_thousand_sep(versement_PER_input))
    col2_vers_PER.metric("Impot avec versement", format_space_thousand_sep(IR_versement_PER), "")
    col3_vers_PER.metric("Economie d'impots", format_space_thousand_sep(IR_one_shot - IR_versement_PER), "")
    col1_vers_PER.metric("Effort d'Epargne", format_space_thousand_sep(versement_PER_input-(IR_one_shot - IR_versement_PER)))
        
    
    fig_PER_versement = px.line(df_income_tax_evol, x='Revenu', y=['IR'])

    if IR_one_shot is not None:
        fig_PER_versement.add_scatter(x=[salary * .9],
                y=[IR_one_shot],
                text=format_space_thousand_sep(IR_one_shot),
                marker=dict(
                    color='red',
                    size=10
                ),
                name='IR actuel')
    fig_PER_versement.add_scatter(x=[salary * .9 - versement_PER_input],
                                y=[IR_versement_PER],
                                text=format_space_thousand_sep(IR_versement_PER),
                                marker=dict(
                                    color='green',
                                    size=10
                                    ),
                                name='IR avec versement')
    st.plotly_chart(fig_PER_versement)
        
with tab_PER_effet:
    if(button_run):
        versement_PER_effect_fragment()

        


    




    
    
    
    
    



    
    
    
