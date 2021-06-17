try:
	import pptx
except ModuleNotFoundError as e:
	Warning('Please install python-pptx for Powerpoint reports. - conda install python-pptx')
	raise e
from tigerml.core.utils import slugify
prs = pptx.Presentation()


class slide_layouts:
	def __init__(self):
		for layout in prs.slide_layouts:
			setattr(self, slugify(layout.name), layout)
		self.list = [slugify(l.name) for l in prs.slide_layouts]


layouts = slide_layouts()

from .lib import create_ppt_report
from .Report import PptReport, Section
from .Slide import SlideComponent, Slide

