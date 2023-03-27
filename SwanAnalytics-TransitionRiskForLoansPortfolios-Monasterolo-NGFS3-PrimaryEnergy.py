import streamlit as st
import pandas as pd
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

EjDeltaRatio=1
Chi=0.1
RecoveryRate=0.5
year0=2010
InitProbDefault=0.2 #determines the maximum that the loan value can gain in case of a positive shock

Row1 = st.container()
Row2 = st.container()
Row3 = st.container()
loanvalue = st.container()

@st.cache_data
def get_data():
    conn = pyam.iiasa.Connection()
    conn.connect('ngfs_phase_3')

    df = conn.query(
        model='*GLOBIOM*',
        region=['*France*','*Germany*'],
        variable='*Primary Energy*'
    )
    
    dff=df.data
    dff=dff[dff['variable'].str.contains('Price') == False]
    dff=dff[dff['year']>=year0]
    dff=dff[(dff['scenario'].str.contains('Below')==True) | (dff['scenario'].str.contains('Current')==True) | (dff['scenario'].str.contains('Delayed')==True)]

    # pivot the primary energy source columns
    dffp = dff.pivot_table(
        values='value',
        index=['scenario','model', 'region','year', 'unit'],
        columns='variable'
        )

    # set the indexes as new columns
    dffp.reset_index(inplace=True)

    #drop the name of the set of columns
    dffp.columns.name=None
    
    return dffp

dffp=get_data()

sectors_columns=['Primary Energy|Biomass', 'Primary Energy|Coal', 'Primary Energy|Gas','Primary Energy|Geothermal', 'Primary Energy|Hydro','Primary Energy|Nuclear', 'Primary Energy|Oil','Primary Energy|Solar','Primary Energy|Wind']
sectors=['Biomass', 'Coal','Gas','Geothermal','Hydro', 'Nuclear', 'Oil','Solar','Wind']
MarketShareSectorsColumns=['MarketShare'+sectors[i] for i in range(len(sectors))]
InitialMarketShareSectorsColumns=['Initial'+MarketShareSectorsColumns[i] for i in range(len(sectors))]
ShocksSectorsColumns=['Shocks'+sectors[i] for i in range(len(sectors))]
NewValueOfLoanSectorsColumns=['NewValueLoan'+sectors[i] for i in range(len(sectors))]


# add a new column for the total primary energy, as the sum of the different primary energy of all the sectors
dffp['PrimaryEnergyCalculated'] = dffp[sectors_columns].sum(axis=1)

#add to the inition pandas dataframe, columns for  market shares for each of the sectors of the sector list.
for i in range(len(sectors)):
    dffp[MarketShareSectorsColumns[i]] = dffp[sectors_columns[i]].div(dffp['PrimaryEnergyCalculated'])


# add columns for initial market size for primary energy, depending on the country. This will be used to compute the shocks
for i in range(len(sectors)):
  region_list = dffp['region'].unique()
  
  #extract the list of initial market shares, it has as many elements as there are regions
  initial_market_shares_list = []
  for region in region_list:
    row_of_interest = dffp[(dffp['scenario'].str.contains('Below')==True) & (dffp['year']==year0) & (dffp['region'] == region)]
    initial_market_shares_list.append(row_of_interest[MarketShareSectorsColumns[i]].iloc[0])

  #create a function, you give it the region, it gives you the initial market share in that region (for a given sector)
  mapping=dict(zip(region_list,initial_market_shares_list))

  #create a new column (for a given sector), which gives the initial market shares, depending on the region
  dffp[InitialMarketShareSectorsColumns[i]]=dffp['region'].apply(lambda x: mapping[x])    


# compute the shocks, add them as columns, a column for each sector
for i in range(len(sectors)):
  dffp[ShocksSectorsColumns[i]] = (dffp[MarketShareSectorsColumns[i]]-dffp[InitialMarketShareSectorsColumns[i]]).div(dffp[InitialMarketShareSectorsColumns[i]])


#compute the market shares per sector at year0
InitialMarketSharesDataFrame=dffp[['scenario','year', 'region']+InitialMarketShareSectorsColumns][(dffp['year']==year0) & (dffp['scenario'].str.contains('Below')==True)]
UnpivotedInitialMarketSharesDataFrame =  pd.melt(InitialMarketSharesDataFrame, id_vars=['scenario','year', 'region'])


#Shocks matrix
ShocksMatrix=dffp[['scenario','year', 'region']+ShocksSectorsColumns]




with Row1:
    st.title('Transition risk for a loan portfolio - Climate stress test')
    #st.text('')

with Row2:
    st.header('Market shares of primary energy sources')
    
    Row2Col1, Row2Col2, Row2Col3 = st.columns([1,2,2])
    Row2Col1.markdown('* **Select your variables** ')
    TheRegion = Row2Col1.selectbox('Choose the region:', options=dffp['region'].unique())
    TheScenario= Row2Col1.selectbox('Choose the region:', options=dffp['scenario'].unique())
    TheSector=Row2Col1.selectbox("Choose Sector for which you want to  plot the shocks:", options=['Biomass', 'Coal','Gas','Geothermal','Hydro','Solar','Wind', 'Nuclear', 'Oil'])
    Chi = 0.01*Row2Col1.slider('Chi as a percentage', min_value=0, max_value=100, value=100, step=10)

    Row2Col2.markdown('* **This graph:** shows the projections of market shares of primary energy sorces for the coming years for the selected regions and scenarios ')
    StackedMarketSharesPlot = dffp[(dffp['region'] == TheRegion) & (dffp['scenario'] == TheScenario)][sectors_columns].plot.area()
    StackedMarketSharesPlot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    StackedMarketSharesPlot.set_ylabel('Energy consuption EJ/yr')
    StackedMarketSharesPlot.set_xlabel('Year')
    StackedMarketSharesPlot.set_title(str(TheRegion)+' - '+str(TheScenario))
    Row2Col2.pyplot(StackedMarketSharesPlot.figure)
    
    Row2Col3.markdown('* **This graph:** shows the marketshare of each energy sector in year '+str(year0) )
    InitialMarketSharesPlot = sns.catplot(data=UnpivotedInitialMarketSharesDataFrame, y='variable',x='value', kind='bar',hue='region', aspect=8/5)
    InitialMarketSharesPlot.set_axis_labels('Energy consuption EJ/yr', '')
    Row2Col3.pyplot(InitialMarketSharesPlot)
    
    
    
    

with Row3:
    
    Row3Col1, Row3Col2, Row3Col3, Row3Col4 = st.columns([2,1,1,2])
    
    

    Row3Col1.header('The shocks')
    
    SchocksPlot = sns.relplot(data=dffp[dffp['region']==TheRegion], kind='line', x='year', y='Shocks'+TheSector, hue='scenario', height=5, aspect=8/5)
    Row3Col1.pyplot(SchocksPlot)

    Row3Col2.header('Choose your portfolio')


    with Row3Col2:
        Row3Col2Col1, Row3Col2Col2 = st.columns(2)
    
    FaceValuesOfLoans1=[0]*len(sectors)

    for i in range(len(sectors)):
        if i < 0.5*len(sectors):
            FaceValuesOfLoans1[i] = Row3Col2Col1.slider(sectors[i], min_value=0, max_value=1000, value=1000, step=10)
        else:
            FaceValuesOfLoans1[i] = Row3Col2Col2.slider(sectors[i], min_value=0, max_value=1000, value=1000, step=10)
    
    FaceValuesOfLoans2={'Sectors':sectors, 'Face Values of Loans':FaceValuesOfLoans1}
    FaceValuesOfLoans=pd.DataFrame(FaceValuesOfLoans2)

    #ChangeInDefaultProbMatrix
    ChangeInDefaultProbMatrix =ShocksMatrix.copy()
    ChangeInDefaultProbMatrix[ShocksSectorsColumns] = -EjDeltaRatio*Chi*ShocksMatrix[ShocksSectorsColumns]
    ChangeInDefaultProbMatrix[ShocksSectorsColumns] = ChangeInDefaultProbMatrix[ShocksSectorsColumns].apply(lambda x: [y if y >= -InitProbDefault else -InitProbDefault for y in x])

    # Compute the change in values of loans. This should be modified since in our probability matrix we have probability changes of 40!, we did not take into account the fact that if the shock is very positive, then the lender will not return more than what we took.
    ChangeInValueOfLoans1=-np.dot(ChangeInDefaultProbMatrix[ShocksSectorsColumns].to_numpy(), FaceValuesOfLoans['Face Values of Loans'].to_numpy())*(1-RecoveryRate)
    ChangeInValueOfLoans2={'scenario':ChangeInDefaultProbMatrix['scenario'], 'year': ChangeInDefaultProbMatrix['year'],'region': ChangeInDefaultProbMatrix['region'],  'Change in value of loans':ChangeInValueOfLoans1}
    ChangeInValueOfLoans=pd.DataFrame(ChangeInValueOfLoans2)
    ChangeInValueOfLoans['New value of loans']=ChangeInValueOfLoans['Change in value of loans']+FaceValuesOfLoans['Face Values of Loans'].sum()

    
    Row3Col4.header('Simulation for the value of a loan')
    
    LoanValueSimulation=sns.relplot(data=ChangeInValueOfLoans[ChangeInValueOfLoans['region']==TheRegion], kind='line', x='year', y='New value of loans', hue='scenario', col='region', height=5, aspect=8/5)
    Row3Col3.pyplot(LoanValueSimulation)


    ChangeInValueOfLoansMatrix=ChangeInDefaultProbMatrix.copy()
    for i in range(len(sectors)):
        ChangeInValueOfLoansMatrix[ShocksSectorsColumns[i]]=-ChangeInValueOfLoansMatrix[ShocksSectorsColumns[i]]*FaceValuesOfLoans['Face Values of Loans'][i]*(1-RecoveryRate)

    NewValueOfLoansMatrix=pd.DataFrame()
    NewValueOfLoansMatrix['scenario']=ChangeInValueOfLoansMatrix['scenario']
    NewValueOfLoansMatrix['year']=ChangeInValueOfLoansMatrix['year']
    NewValueOfLoansMatrix['region']=ChangeInValueOfLoansMatrix['region']
    for i in range(len(sectors)):
        NewValueOfLoansMatrix[NewValueOfLoanSectorsColumns[i]]=FaceValuesOfLoans['Face Values of Loans'][i]+ChangeInValueOfLoansMatrix[ShocksSectorsColumns[i]]

    NewValueOfLoansMatrix['TotalNewValueLoan']=NewValueOfLoansMatrix[NewValueOfLoanSectorsColumns].sum(axis=1)

    StackedNewValueOfLoansPlot = NewValueOfLoansMatrix[(NewValueOfLoansMatrix['region'] == TheRegion) & (NewValueOfLoansMatrix['scenario'] == TheScenario)][NewValueOfLoanSectorsColumns].plot.area()
    StackedNewValueOfLoansPlot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    StackedNewValueOfLoansPlot.set_ylabel('Loan value')
    StackedNewValueOfLoansPlot.set_xlabel('Year')
    StackedNewValueOfLoansPlot.set_title(str(TheRegion)+' - '+str(TheScenario))
    Row3Col4.pyplot(StackedNewValueOfLoansPlot.figure)
