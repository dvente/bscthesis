# LaTeX Makefile
FILE = thesis
all: $(FILE).pdf

.PHONY: clean

clean:
	\rm $(FILE).aux $(FILE).blg $(FILE).bbl $(FILE).log $(FILE).synctex.gz $(FILE).toc

$(FILE).pdf: 
	$(FILE).tex
	pdflatex $(FILE)
	pdflatex $(FILE)
	bibtex $(FILE)
	pdflatex $(FILE)
	pdflatex $(FILE)