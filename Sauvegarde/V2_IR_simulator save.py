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

step = 100
round_num = -2

def round_to(num, round_num=round_num):
    return int(np.round(num, round_num))

def format_space_thousand_sep(num, trailing=" €"):
    return '{:,}'.format(np.round(num, 0)).replace(',', ' ') + trailing

def add_bracket_lines(fig, df_simulation, bareme_annee_simulation):
    """
    Adds vertical lines to the graph at the points where the official TMI bracket changes.
    This method uses the simulation data itself to find the thresholds, ensuring perfect alignment
    with the curve, as it accounts for all complex interactions (10% deduction, CSG, décote, etc.).
    """
    # Find the rows where the official TMI bracket (ir_tranche) changes compared to the previous row.
    tmi_bracket_changes = df_simulation['ir_tranche'].diff().ne(0)
    change_points = df_simulation[tmi_bracket_changes]

    # Iterate over the points where the TMI changes and draw a line.
    for _, row in change_points.iterrows():
        bracket_index = int(row['ir_tranche'])
        rbg_threshold = row['Revenu']
        # Only draw lines for actual taxable brackets (index > 0)
        if bracket_index > 0:
            official_rate = bareme_annee_simulation.rates[bracket_index]
            fig.add_vline(x=rbg_threshold, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"TMI {int(official_rate*100)}%", annotation_position="top right", annotation_font_size=10)
    return fig


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
utilisation_reforme= False
if utilisation_reforme:
    legislation_france = FranceTaxBenefitSystem()
    legislation_reforme = MaReforme(legislation_france)
else:
    legislation_reforme = FranceTaxBenefitSystem()

with st.sidebar:
    input_annee_simulation = st.selectbox("Annee de perception des revenus", (2024, 2023, 2022, 2021, 2020))

    input_nb_adultes = st.selectbox("Nombre d'adultes", (1, 2))
    input_parent_isole = False
    if input_nb_adultes == 1:
        input_parent_isole = st.checkbox('Parent isolé?')
    input_nb_enfants = st.selectbox("Nombre d'enfants à charge", (0, 1, 2, 3, 4, 5))
    input_nb_alternes = st.selectbox("Nombre d'enfants en garde alternée", (0, 1, 2, 3, 4, 5))
    col1, col2 = st.columns(2, vertical_alignment='bottom')
    input_salary = col1.number_input("Salaire annuel net", value=50000, step=step)
    button_run = col2.button("Calcul", icon="😰", use_container_width=True)
    input_plafond_PER = st.number_input("Plafond PER", value=16000)
    with st.expander("Options"):
        input_salary_max_simu = st.number_input("Salaire max simulation", value = input_salary, step=step)

def run_all_simulations(annee_simulation, nb_adultes, parent_isole, nb_enfants, nb_alternes, salary, plafond_PER, salary_max_simu, legislation_reforme):
    """
    Performs all OpenFisca calculations and returns a dictionary of results.
    """
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

    CASE['familles'] = {'famille1': {
        'parents': ['parent' + str(i+1) for i in range(nb_adultes)],
        'enfants': ['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]
    }}
    CASE['menages'] = {'menage1': {
        'personne_de_reference': ['parent1'],
        'enfants':['enfant' + str(i+1) for i in range(nb_enfants + nb_alternes)]
    }}
    if nb_adultes > 1:
        CASE['menages']['menage1']['conjoint'] = ['parent2']
    
    #length_simu = int(np.round(salary * 1.2, round_num))
    length_simu = round_to(max(salary * 1.2, salary_max_simu))
    length_simu_n = int(length_simu/step)
    
    CASE['axes']= [[{'count':length_simu_n, 'name':'salaire_imposable', 'min':0, 'max':length_simu,
                     'period': annee_simulation}]]
        
    # Simulation IR
    n_reshape = nb_adultes + nb_alternes + nb_enfants
    simulation_builder = SimulationBuilder()
    simulation_evol = simulation_builder.build_from_entities(legislation_reforme, CASE)    
    bareme_annee_simulation = legislation_reforme.parameters.impot_revenu.bareme_ir_depuis_1945.bareme(str(annee_simulation))
    
    nb_part_one_shot = simulation_evol.calculate('nbptr', str(annee_simulation))[0]

    # The axis applies the same RBG to each individual. The foyer's RBG is the sum of declarants' RBG.
    sdb = simulation_evol.calculate_add('salaire_imposable', annee_simulation)
    # Reshape by step and individual, then sum the RBG of the declarants (the first nb_adultes individuals)
    salaire_foyer = sdb.reshape(int(length_simu_n), n_reshape)[:, :nb_adultes].sum(axis=1)
    
    # Foyer-level variables are broadcast to declarants. We reshape and take the value from the first person.
    income_tax_evol = simulation_evol.calculate('ip_net', annee_simulation).reshape(int(length_simu_n), nb_adultes)[:, 0]
    
    avantage_qf_evol = -simulation_evol.calculate('avantage_qf', str(annee_simulation)).reshape(int(length_simu_n), nb_adultes)[:, 0]
    
    TMI_evol = simulation_evol.calculate('ir_taux_marginal', str(annee_simulation)).reshape(int(length_simu_n), nb_adultes)[:, 0]
    ir_tranche_evol = simulation_evol.calculate('ir_tranche', str(annee_simulation)).reshape(int(length_simu_n), nb_adultes)[:, 0]
    
    taux_moyen_imposition_evol = simulation_evol.calculate('taux_moyen_imposition', str(annee_simulation)).reshape(int(length_simu_n), nb_adultes)[:, 0]
    
    IR_ss_qf_evol = simulation_evol.calculate('ir_ss_qf', str(annee_simulation)).reshape(int(length_simu_n), nb_adultes)[:, 0]
    
    decote_evol = simulation_evol.calculate('decote_gain_fiscal', str(annee_simulation)).reshape(int(length_simu_n), nb_adultes)[:, 0]
    
    df_income_tax_evol = pd.DataFrame.from_dict({'Revenu': salaire_foyer, 'IR': income_tax_evol,
                                                 'TMI': TMI_evol,
                                                 'Taux moyen d imposition' :taux_moyen_imposition_evol,
                                                 'IR sans QF' : IR_ss_qf_evol,
                                                 'Reduction QF': avantage_qf_evol, 
                                                 'ir_tranche': ir_tranche_evol,
                                                 'Decote': decote_evol})
    
    df_income_tax_evol['Revenu'] = df_income_tax_evol['Revenu'].apply(round_to)#.apply(int)
    ss['df_income_tax_evol'] = df_income_tax_evol

    df_one_shot = df_income_tax_evol[df_income_tax_evol['Revenu'] == round_to(salary)]
    
    # Get the current official TMI bracket index to determine the optimization target
    current_official_tmi_bracket_index = 0
    if not df_one_shot.empty:
        current_official_tmi_bracket_index = int(df_one_shot['ir_tranche'].values[0])

    # Montant optimal PER
    CASE_PER = CASE.copy()
    CASE_PER['individus']['parent1']['salaire_imposable'] = {annee_simulation: salary}
    length_simu_PER = int(salary)
    CASE_PER['axes']= [[{'count':length_simu_PER, 'name':'f6rs', 'min':0, 'max':length_simu_PER, 'period': annee_simulation}]]
    
    n_reshape = nb_adultes + nb_alternes + nb_enfants
    
    simulation_builder = SimulationBuilder()
    simulation_PER = simulation_builder.build_from_entities(legislation_reforme, CASE_PER)
    
    versement_PER = simulation_PER.calculate_add('f6rs', annee_simulation)
    # The x-axis is the total PER payment from declarants
    versement_PER = versement_PER.reshape(int(length_simu_PER), n_reshape)[:, :nb_adultes].sum(axis=1)
    
    income_tax_PER = simulation_PER.calculate('ip_net', annee_simulation).reshape(int(length_simu_PER), nb_adultes)[:, 0]
    TMI_PER = simulation_PER.calculate('ir_taux_marginal', annee_simulation).reshape(int(length_simu_PER), nb_adultes)[:, 0]
    ir_tranche_PER = simulation_PER.calculate('ir_tranche', annee_simulation).reshape(int(length_simu_PER), nb_adultes)[:, 0]
    
    df_income_tax_PER = pd.DataFrame.from_dict({'Versement_PER': versement_PER, 
                                                'IR': income_tax_PER,
                                                'TMI': TMI_PER,
                                                'ir_tranche': ir_tranche_PER})
    nb_part_PER = simulation_PER.calculate('nbptr', str(annee_simulation))[0]

    # Find the next lower official TMI bracket to target for PER optimization.
    # We look for the largest bracket index that is less than the current one.
    lower_official_tmi_brackets = df_income_tax_PER['ir_tranche'][df_income_tax_PER['ir_tranche'] < current_official_tmi_bracket_index]
    target_official_tmi_bracket_index = 0
    if not lower_official_tmi_brackets.empty:
        target_official_tmi_bracket_index = int(lower_official_tmi_brackets.max())
    # Find the first row (smallest contribution) that reaches this lower official bracket
    df_optimal_PER = df_income_tax_PER[df_income_tax_PER['ir_tranche'] == target_official_tmi_bracket_index].head(1)
    
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
        
    return {
        "df_income_tax_evol": df_income_tax_evol,
        "df_income_tax_PER": df_income_tax_PER,
        "df_one_shot": df_one_shot,
        "nb_part_one_shot": nb_part_one_shot,
        "bareme_annee_simulation": bareme_annee_simulation,
        "versement_optimal_PER": versement_optimal_PER,
        "impot_PER": impot_PER,
        "versement_PER_TMI": versement_PER_TMI,
        "length_simu": length_simu,
    }

def display_one_shot_tab(results, salary):
    """
    Displays the content for the 'Calcul de l'IR' tab.
    """
    df_one_shot = results['df_one_shot']
    if df_one_shot.empty:
        st.error("Impossible de calculer l'impôt pour le salaire spécifié. Essayez d'augmenter le salaire max de la simulation.")
        return

    nb_part_one_shot = results['nb_part_one_shot']
    bareme_annee_simulation = results['bareme_annee_simulation']
    df_income_tax_evol = results['df_income_tax_evol']
    length_simu = results['length_simu']

    try:
        Revenu_one_shot = salary # On affiche le revenu saisi, pas celui de l'axe qui est arrondi
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

        st.divider()
        with st.expander("Afficher les barèmes et seuils"):
            col_bareme1, col_bareme2 = st.columns(2)
            thresholds = bareme_annee_simulation.thresholds
            rates = bareme_annee_simulation.rates

            with col_bareme1:
                st.subheader("Barème IR (pour 1 part)")
                data = []
                for i in range(len(thresholds)):
                    lower_bound = thresholds[i]
                    rate = rates[i]
                    lower_bound_str = format_space_thousand_sep(lower_bound, trailing="")
                    if i < len(thresholds) - 1:
                        upper_bound = thresholds[i+1] - 1
                        upper_bound_str = format_space_thousand_sep(upper_bound, trailing="")
                    else:
                        upper_bound_str = "et plus"
                    data.append({
                        'De (RNI/part)': lower_bound_str,
                        'À (RNI/part)': upper_bound_str,
                        'Taux (%)': f"{int(rate * 100)}",
                    })
                df_brackets = pd.DataFrame(data)
                st.dataframe(df_brackets, hide_index=True, use_container_width=True)

            with col_bareme2:
                st.subheader("Seuils TMI du foyer (RBG)")
                # We derive the thresholds directly from the simulation to ensure consistency with the graph and official rates.
                tmi_bracket_changes = df_income_tax_evol['ir_tranche'].diff().ne(0)
                change_points = df_income_tax_evol[tmi_bracket_changes]

                data_foyer = []
                # Iterate over change points, skipping the first (0% bracket)
                for _, row in change_points.iloc[1:].iterrows():
                    bracket_index = int(row['ir_tranche'])
                    # Only add rows for actual taxable brackets (index > 0)
                    if bracket_index > 0:
                        official_rate = bareme_annee_simulation.rates[bracket_index]
                    data_foyer.append({
                        'Taux Marginal': f"{int(official_rate * 100)} %",
                        'À partir de (RBG)': format_space_thousand_sep(row['Revenu']),
                    })
                df_brackets_foyer = pd.DataFrame(data_foyer)
                st.dataframe(df_brackets_foyer, hide_index=True, use_container_width=True)
        st.divider()
        
        fig = px.line(df_income_tax_evol, x='Revenu', y=['IR'])
        add_bracket_lines(fig, df_income_tax_evol, bareme_annee_simulation)

        if not df_one_shot.empty:
            fig.add_scatter(x=[salary],
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
            yaxis_ticksuffix = '€',
            xaxis_range=[0, length_simu])
        st.plotly_chart(fig)
    except (IndexError, KeyError):
        st.error("Erreur lors de la récupération des données de simulation. Vérifiez les paramètres.")

def display_per_optim_tab(results, salary, plafond_PER):
    """
    Displays the content for the 'Versement Optimal PER' tab.
    """
    df_one_shot = results['df_one_shot']
    if df_one_shot.empty:
        return # Error already shown in the first tab

    IR_one_shot = df_one_shot['IR'].values[0]
    versement_optimal_PER = results['versement_optimal_PER']
    impot_PER = results['impot_PER']
    versement_PER_TMI = results['versement_PER_TMI']
    df_income_tax_evol = results['df_income_tax_evol']
    df_income_tax_PER = results['df_income_tax_PER']
    bareme_annee_simulation = results['bareme_annee_simulation']
    length_simu = results['length_simu']

    try:
        versement_PER_preco = min(versement_optimal_PER, plafond_PER)
        
        col1_PER, col2_PER, col3_PER = st.columns(3)
        col1_PER.metric("Versement Optimal PER", format_space_thousand_sep(versement_PER_preco))
        col2_PER.metric("Impot avec versement", format_space_thousand_sep(impot_PER), "")
        col3_PER.metric("Economie d'impots", format_space_thousand_sep(IR_one_shot - impot_PER), "")
        col1_PER.metric("Versement Optimal TMI", format_space_thousand_sep(versement_PER_TMI))
        
        fig_PER_evol = px.line(df_income_tax_evol, x='Revenu', y=['IR'])
        add_bracket_lines(fig_PER_evol, df_income_tax_evol, bareme_annee_simulation)


        if not df_one_shot.empty:
            fig_PER_evol.add_scatter(x=[salary],
                    y=[IR_one_shot],
                    text=format_space_thousand_sep(IR_one_shot),
                    marker=dict(
                        color='red',
                        size=10
                    ),
                    name='IR actuel')

        # Find the point on the original tax curve that corresponds to the tax with PER payment
        # This ensures the green dot is perfectly on the blue curve.
        equivalent_row_optimal = df_income_tax_evol.iloc[(df_income_tax_evol['IR'] - impot_PER).abs().argmin()]
        equivalent_revenu_optimal = equivalent_row_optimal['Revenu']
        equivalent_ir_optimal = equivalent_row_optimal['IR']

        fig_PER_evol.add_scatter(x=[equivalent_revenu_optimal],
                                 y=[equivalent_ir_optimal],
                                 text=format_space_thousand_sep(impot_PER), # Show the actual tax value
                                 marker=dict(
                                     color='green',
                                     size=10
                                     ),
                                 name='IR avec versement')
        fig_PER_evol.update_layout(
            title="Evolution Impôt sur le revenu avec versement PER optimal",
            xaxis_title="Revenu net annuel",
            yaxis_title="Montant de l'IR",
            xaxis_ticksuffix = '€',
            yaxis_ticksuffix = '€',
            xaxis_range=[0, length_simu])
        st.plotly_chart(fig_PER_evol)
        
        fig_PER = px.line(df_income_tax_PER, x='Versement_PER', y=['IR']) # Removed df_income_tax_PER.copy()
        fig_PER.update_layout(xaxis_range=[0, plafond_PER])
        st.plotly_chart(fig_PER)

        fig_tmi_PER = px.line(df_income_tax_PER, x='Versement_PER', y=['TMI']) # Removed df_income_tax_PER.copy()
        fig_tmi_PER.update_layout(xaxis_range=[0, plafond_PER])
        st.plotly_chart(fig_tmi_PER)
    except (IndexError, KeyError):
        st.error("Erreur lors de l'affichage de l'optimisation PER.")

def display_per_effect_tab(results, salary, plafond_PER):
    """
    Displays the content for the 'Effet Versement PER' tab.
    """
    df_one_shot = results['df_one_shot']
    if df_one_shot.empty:
        return # Error already shown in the first tab

    IR_one_shot = df_one_shot['IR'].values[0]
    df_income_tax_PER = results['df_income_tax_PER']
    df_income_tax_evol = results['df_income_tax_evol']
    bareme_annee_simulation = results['bareme_annee_simulation']
    length_simu = results['length_simu']

    versement_PER_input = st.slider("Versement PER", min_value=0, max_value=int(plafond_PER), value=0, step=100)


    df_versement_PER = df_income_tax_PER[df_income_tax_PER['Versement_PER'] == versement_PER_input].head(1)
    IR_versement_PER = df_versement_PER['IR'].values[0]
    
    col1_vers_PER, col2_vers_PER, col3_vers_PER = st.columns(3)
    col1_vers_PER.metric("Impot sans versement", format_space_thousand_sep(IR_one_shot), "")
    col2_vers_PER.metric("Impot avec versement", format_space_thousand_sep(IR_versement_PER), "")
    col3_vers_PER.metric("Economie d'impots", format_space_thousand_sep(IR_one_shot - IR_versement_PER), "")
    col1_vers_PER.metric("Versement PER", format_space_thousand_sep(versement_PER_input))
    col2_vers_PER.metric("Effort d'Epargne", format_space_thousand_sep(versement_PER_input-(IR_one_shot - IR_versement_PER)))
        
    
    fig_PER_versement = px.line(df_income_tax_evol, x='Revenu', y=['IR'])
    add_bracket_lines(fig_PER_versement, df_income_tax_evol, bareme_annee_simulation)

    if not df_one_shot.empty:
        fig_PER_versement.add_scatter(x=[salary],
                y=[IR_one_shot],
                text=format_space_thousand_sep(IR_one_shot),
                marker=dict(
                    color='red',
                    size=10
                ),
                name='IR actuel')

    # Find the point on the original tax curve that corresponds to the tax with PER payment
    # This ensures the green dot is perfectly on the blue curve.
    equivalent_row_slider = df_income_tax_evol.iloc[(df_income_tax_evol['IR'] - IR_versement_PER).abs().argmin()]
    equivalent_revenu_slider = equivalent_row_slider['Revenu']
    equivalent_ir_slider = equivalent_row_slider['IR']

    fig_PER_versement.add_scatter(x=[equivalent_revenu_slider],
                                y=[equivalent_ir_slider],
                                text=format_space_thousand_sep(IR_versement_PER), # Show the actual tax value
                                marker=dict(
                                    color='green',
                                    size=10
                                    ),
                                name='IR avec versement')
    fig_PER_versement.update_layout(
        title="Evolution Impôt sur le revenu avec versement PER",
        xaxis_title="Revenu net annuel",
        yaxis_title="Montant de l'IR",
        xaxis_ticksuffix = '€',
        yaxis_ticksuffix = '€',
        xaxis_range=[0, length_simu])
    st.plotly_chart(fig_PER_versement)

    # --- Graphique de l'économie d'impôt ---
    df_eco = df_income_tax_PER.copy()
    df_eco["Economie_IR"] = IR_one_shot - df_eco['IR']

    # Filtrer les données jusqu'au plafond PER
    df_eco_filtered = df_eco[df_eco['Versement_PER'] <= plafond_PER]

    fig_eco_per = px.line(df_eco_filtered, x='Versement_PER', y="Economie_IR")

    # Add the green dot for the current slider selection
    economie_ir_slider = IR_one_shot - IR_versement_PER
    fig_eco_per.add_scatter(
        x=[versement_PER_input],
        y=[economie_ir_slider],
        text=format_space_thousand_sep(economie_ir_slider),
        marker=dict(
            color='green',
            size=10
        ),
        name="Economie pour ce versement",
        showlegend=False
    )
    fig_eco_per.update_layout(
        title="Economie d'impôt en fonction du versement PER",
        xaxis_title="Versement PER",
        yaxis_title="Economie d'impôt",
        xaxis_ticksuffix = '€',
        yaxis_ticksuffix = '€')
    st.plotly_chart(fig_eco_per)

# --- Main App Logic ---

if button_run:
    with st.spinner("Calculs en cours..."):
        ss.simulation_results = run_all_simulations(
            annee_simulation=input_annee_simulation,
            nb_adultes=input_nb_adultes,
            parent_isole=input_parent_isole,
            nb_enfants=input_nb_enfants,
            nb_alternes=input_nb_alternes,
            salary=input_salary,
            plafond_PER=input_plafond_PER,
            salary_max_simu=input_salary_max_simu,
            legislation_reforme=legislation_reforme
        )

if 'simulation_results' in ss:
    tab_one_shot, tab_PER_effet, tab_PER_optim = st.tabs(["Calcul de l'IR", "Effet Versement PER","Versement Optimal PER"])

    with tab_one_shot:
        display_one_shot_tab(ss.simulation_results, input_salary)

    with tab_PER_optim:
        display_per_optim_tab(ss.simulation_results, input_salary, input_plafond_PER)

    with tab_PER_effet:
        display_per_effect_tab(ss.simulation_results, input_salary, input_plafond_PER)

        


    




    
    
    
    
    



    
    
    
