import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("./data/udemy_courses.csv")

# Select features and target variable
features = ["price", "num_reviews", "num_lectures", "content_duration"]
target = "num_subscribers"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the entire dataset for visualization
df["predicted_subscribers"] = model.predict(df[features])

app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(
    children=[
        html.H1(children="Course Analytics Dashboard with Prediction"),
        # Visualization 1: Number of Subscribers by Subject
        dcc.Graph(
            id="subscribers-by-subject",
            figure=px.bar(
                df,
                x="subject",
                y="num_subscribers",
                title="Number of Subscribers by Subject",
            ),
        ),
        # Visualization 2: Distribution of Course Levels
        dcc.Graph(
            id="course-level-distribution",
            figure=px.pie(df, names="level", title="Distribution of Course Levels"),
        ),
        # Visualization 3: Price Distribution
        dcc.Graph(
            id="price-distribution",
            figure=px.histogram(df, x="price", title="Price Distribution"),
        ),
        # Visualization 4: Number of Lectures by Subject
        dcc.Graph(
            id="lectures-by-subject",
            figure=px.scatter(
                df, x="subject", y="num_lectures", title="Number of Lectures by Subject"
            ),
        ),
        # Visualization 5: Predicted vs Actual Number of Subscribers
        dcc.Graph(
            id="predicted-vs-actual-subscribers",
            figure=px.scatter(
                df,
                x="num_subscribers",
                y="predicted_subscribers",
                title="Predicted vs Actual Subscribers",
            ),
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
