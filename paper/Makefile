# Makefile for sn-article.tex (Springer Nature, IJIT)
#
# Usage:
#   make            # build the PDF
#   make clean      # remove auxiliary files
#   make distclean  # also remove the PDF
#
# Prerequisites:
#   - pdflatex
#   - bibtex
#   - the Springer Nature class file `sn-jnl.cls` and bibliography style
#     `sn-basic.bst`. Download both from
#     https://www.springernature.com/gp/authors/campaigns/latex-author-support
#     and place them in this directory (or in your TEXMF tree).
#   - the figures/ directory (already populated by the build script).

PAPER     := sn-article
TEX       := pdflatex
BIB       := bibtex
TEXFLAGS  := -interaction=nonstopmode -halt-on-error
FIGS      := $(wildcard figures/*.png)

.PHONY: all clean distclean view check

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib $(FIGS)
	$(TEX) $(TEXFLAGS) $(PAPER).tex
	$(BIB) $(PAPER)
	$(TEX) $(TEXFLAGS) $(PAPER).tex
	$(TEX) $(TEXFLAGS) $(PAPER).tex
	@echo ""
	@echo "Built $(PAPER).pdf — page count:"
	@pdfinfo $(PAPER).pdf 2>/dev/null | grep Pages || true

view: $(PAPER).pdf
	@if command -v xdg-open >/dev/null; then xdg-open $(PAPER).pdf; \
	elif command -v open    >/dev/null; then open    $(PAPER).pdf; \
	else echo "No PDF viewer found"; fi

# Sanity check: confirm every cited reference exists in references.bib
check:
	@grep -oE '\\cite\{[^}]+\}' $(PAPER).tex | \
	  sed -E 's/\\cite\{//; s/\}//' | tr ',' '\n' | sort -u > /tmp/cited_keys.txt
	@grep -oE '^@[a-zA-Z]+\{[^,]+' references.bib | \
	  sed -E 's/^@[a-zA-Z]+\{//' | sort -u > /tmp/bib_keys.txt
	@missing=$$(comm -23 /tmp/cited_keys.txt /tmp/bib_keys.txt); \
	if [ -n "$$missing" ]; then \
	  echo "MISSING in references.bib:"; echo "$$missing"; exit 1; \
	else echo "All citations resolve."; fi

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).out $(PAPER).toc $(PAPER).synctex.gz \
	      $(PAPER).fdb_latexmk $(PAPER).fls

distclean: clean
	rm -f $(PAPER).pdf
