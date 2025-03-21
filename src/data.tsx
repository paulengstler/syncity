// Project title
export const title = "SynCity: Training-Free Generation of 3D Worlds"

// Abstract
export const abstract = "We address the challenge of generating 3D worlds from textual descriptions. We propose SynCity, a training- and optimization-free approach, which leverages the geometric precision of pre-trained 3D generative models and the artistic versatility of 2D image generators to create large, high-quality 3D spaces.  While most 3D generative models are object-centric and cannot generate large-scale worlds, we show how 3D and 2D generators can be combined to generate ever-expanding scenes. Through a tile-based approach, we allow fine-grained control over the layout and the appearance of scenes. The world is generated tile-by-tile, and each new tile is generated within its world-context and then fused with the scene. SynCity generates compelling and immersive scenes that are rich in detail and diversity."

export const description = "Generate complex and immersive 3D worlds from text prompts without any training or optimization";

// Authors
export const authors = [
  {
    'name': 'Paul Engstler',
    'institutions': ["University of Oxford"],
    'url': "https://paulengstler.com",
    'tags': ["*"]
  },
  {
    'name': 'Aleksandar Shtedritski',
    'institutions': ["University of Oxford"],
    'url': "https://suny-sht.github.io",
    'tags': ["*"]
  },
  {
    'name': 'Iro Laina',
    'institutions': ["University of Oxford"],
    'url': "http://campar.in.tum.de/Main/IroLaina",
    'tags': []
  },
  {
    'name': 'Christian Rupprecht',
    'institutions': ["University of Oxford"],
    'url': "https://chrirupp.github.io/",
    'tags': []
  },
  {
    'name': 'Andrea Vedaldi',
    'institutions': ["University of Oxford"],
    'url': "https://www.robots.ox.ac.uk/~vedaldi/",
    'tags': []
  },
]

export const author_tags = [
  {
    'name': '*',
    'description': "equal contribution"
  }
]

// Links
export const links = {
  'paper': "https://arxiv.org/abs/2503.16420",
  'github': "https://github.com/paulengstler/syncity",
}

// Acknowledgements
export const acknowledgements = "The authors of this work are supported by ERC 101001212-UNION, AIMS EP/S024050/1, and Meta Research."

// Citations
export const citationId = "engstler2025syncity"
export const citationAuthors = "Paul Engstler and Aleksandar Shtedritski and Iro Laina and Christian Rupprecht and Andrea Vedaldi"
export const citationYear = "2025"
export const citationBooktitle = "Arxiv"