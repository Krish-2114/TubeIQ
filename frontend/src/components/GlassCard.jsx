export default function GlassCard({
  children,
  className = '',
  accent,
  glow = false,
  hover = false,
  onClick,
}) {
  const classes = [
    'glass-card',
    glow && 'card-glow',
    hover && 'card-hover',
    onClick && 'cursor-pointer',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div
      className={classes}
      style={accent ? { borderTop: `3px solid ${accent}` } : undefined}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter') onClick();
            }
          : undefined
      }
    >
      {children}
    </div>
  );
}
