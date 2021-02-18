from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, RadioField, SubmitField
from wtforms.validators import InputRequired
from bail import X_test, y_test, plea_list
import joblib

# SECTION Class for input form
class BailForm(FlaskForm):
    class Meta:
        def render_field(self, field, render_kw):
            render_kw.setdefault('required', True)
            return super().render_field(field, render_kw)
    input_plea_orc = SelectField(choices=plea_list)
    input_race = RadioField('Race', choices=['White', 'Black', 'Hispanic', 'Asian', 'Other'], validators=[InputRequired()])
    input_priors = StringField('Prior Cases', validators=[InputRequired()])
# !SECTION End form class

# NOTE This is where the model for the bail should be loaded
model = joblib.load('bail_model')

app = Flask(__name__)

# NOTE This needs to be here for FlaskForms
app.config['SECRET_KEY'] = '_+aro@(rjwcan9*=l%&(l=cs7mzbh-&5w1g%7)c3*i3ms(vzvf'

@app.route('/', methods=['GET', 'POST'])
def home():
    form = BailForm()

    plea_orc_list = plea_list
    return render_template('index.html', form=form, plea_orc_list=plea_list)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = BailForm()
    # TODO Import unique plea list
    plea_orc_list = plea_list

    # SECTION Race transform
    # TODO Find out what races are transformed as (ex if White: white = 4)

    # !SECTION

    user_input = [[form.input_plea_orc.data, form.input_race.data, form.input_priors.data]]

    # SECTION predict output and show accuracy score
    output = model.predict(user_input)
    score = model.score(X_test, y_test)
    # !SECTION
    return render_template('index.html', prediction_text ='Bail: {}'.format(output), accuracy='Algorithm accuracy: {:.0%}'.format(score), form=form, plea_orc_list=plea_list)

app.run(debug=True)
