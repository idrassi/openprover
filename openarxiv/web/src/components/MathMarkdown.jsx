import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'

/**
 * remark-math converts $...$ to <code class="language-math math-inline">
 * and $$...$$ to <pre><code class="language-math math-display">.
 * We intercept these and render them as raw delimited text so
 * MathJax (loaded via CDN in index.html) can typeset them.
 */
const mathComponents = {
  code({ className, children, ...props }) {
    if (className === 'language-math math-inline') {
      return <span className="math-inline">{`$${children}$`}</span>
    }
    if (className === 'language-math math-display') {
      return <div className="math-display">{`$$${children}$$`}</div>
    }
    return <code className={className} {...props}>{children}</code>
  },
  // Block math gets wrapped in <pre> by default — unwrap it
  pre({ children }) {
    // If the child is our math-display div, just return it directly
    if (children?.props?.className === 'language-math math-display') {
      return mathComponents.code(children.props)
    }
    return <pre>{children}</pre>
  },
}

export default function MathMarkdown({ children, className }) {
  const containerRef = useRef(null)
  const [hasError, setHasError] = useState(false)

  const typeset = useCallback(() => {
    const el = containerRef.current
    if (!el || !window.MathJax?.typesetPromise) return

    window.MathJax.typesetClear?.([el])

    window.MathJax.typesetPromise([el])
      .then(() => {
        const errors = el.querySelectorAll('mjx-merror')
        setHasError(errors && errors.length > 0)
      })
      .catch(() => {
        setHasError(true)
      })
  }, [])

  useEffect(() => {
    if (!children) return

    // MathJax may not be loaded yet (async CDN script)
    if (window.MathJax?.typesetPromise) {
      typeset()
    } else {
      const id = setInterval(() => {
        if (window.MathJax?.typesetPromise) {
          clearInterval(id)
          typeset()
        }
      }, 100)
      return () => clearInterval(id)
    }
  }, [children, typeset])

  if (!children) return null

  return (
    <div ref={containerRef} className={`math-markdown ${className || ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        components={mathComponents}
      >
        {children}
      </ReactMarkdown>
      {hasError && (
        <details className="math-error">
          <summary>Some math failed to render. Original source:</summary>
          <pre className="math-source">{children}</pre>
        </details>
      )}
    </div>
  )
}
