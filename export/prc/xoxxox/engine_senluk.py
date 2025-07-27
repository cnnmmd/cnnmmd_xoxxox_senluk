import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
from xoxxox.shared import Custom

#---------------------------------------------------------------------------

class SenPrc:

  def __init__(self, config="xoxxox/config_senluk_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    nmodel = diccnf["nmodel"]
    self.lenmax = diccnf["lenmax"]
    cnfluk = LukeConfig.from_pretrained(
      nmodel,
      output_hidden_states=True)
    self.omodel = AutoModelForSequenceClassification.from_pretrained(
      nmodel,
      config=cnfluk)
    self.tokenz = AutoTokenizer.from_pretrained(
      nmodel)

  def status(self, config="xoxxox/config_senluk_000", **dicprm):
    diccnf = Custom.update(config, dicprm)

  def infere(self, txtreq):
    inputs = self.tokenz(
      txtreq,
      truncation=True,
      max_length=self.lenmax,
      padding="max_length")
    output = self.omodel(
      torch.tensor(inputs["input_ids"]).unsqueeze(0),
      torch.tensor(inputs["attention_mask"]).unsqueeze(0))
    maxsen = torch.argmax(
      torch.tensor(output.logits))
    numsen = maxsen.item()
    txtsen = str(numsen)
    return txtsen
