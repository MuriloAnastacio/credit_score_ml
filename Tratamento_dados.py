# Missing
df.fillna(df.median(), inplace=True)

# Separação
X = df.drop('default', axis=1)
y = df['default']
