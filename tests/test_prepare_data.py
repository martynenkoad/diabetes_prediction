import prepare_data

def test_alter_smoking_column():
    """"""
    output1 = prepare_data.alter_smoking_column("never")
    assert output1 == "not_smoker"
    output2 = prepare_data.alter_smoking_column("No Info")
    assert output2 == "not_smoker"
    output3 = prepare_data.alter_smoking_column("current")
    assert output3 == "smoker"
    output4 = prepare_data.alter_smoking_column("ever")
    assert output4 == "past_smoker"
    output5 = prepare_data.alter_smoking_column("former")
    assert output5 == "past_smoker"
    output6 = prepare_data.alter_smoking_column("not current")
    assert output6 == "past_smoker"


