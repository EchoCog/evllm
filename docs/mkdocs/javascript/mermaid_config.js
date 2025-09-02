// Mermaid configuration for vLLM documentation
document.addEventListener('DOMContentLoaded', function() {
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      themeVariables: {
        primaryColor: '#2196f3',
        primaryTextColor: '#1565c0',
        primaryBorderColor: '#1976d2',
        lineColor: '#616161',
        secondaryColor: '#f5f5f5',
        tertiaryColor: '#e3f2fd',
        background: '#ffffff',
        mainBkg: '#ffffff',
        secondBkg: '#f5f5f5',
        tertiaryBkg: '#e3f2fd'
      },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      },
      sequence: {
        useMaxWidth: true,
        wrap: true
      },
      gantt: {
        useMaxWidth: true
      },
      class: {
        useMaxWidth: true
      },
      state: {
        useMaxWidth: true
      },
      pie: {
        useMaxWidth: true
      }
    });

    // Re-initialize Mermaid when theme changes (dark/light mode)
    const observer = new MutationObserver(function(mutations) {
      mutations.forEach(function(mutation) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'data-md-color-scheme') {
          const scheme = document.documentElement.getAttribute('data-md-color-scheme');
          const theme = scheme === 'slate' ? 'dark' : 'default';
          
          mermaid.initialize({
            theme: theme,
            startOnLoad: true
          });
          
          // Re-render all mermaid diagrams
          const mermaidElements = document.querySelectorAll('.mermaid');
          mermaidElements.forEach(function(element) {
            element.innerHTML = element.getAttribute('data-original-content') || element.innerHTML;
            if (!element.getAttribute('data-original-content')) {
              element.setAttribute('data-original-content', element.innerHTML);
            }
          });
          
          mermaid.init();
        }
      });
    });
    
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-md-color-scheme']
    });
  }
});