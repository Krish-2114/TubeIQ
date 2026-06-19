export default function SectionHeader({ label, title, subtitle, align = 'left' }) {
  const alignClass =
    align === 'center' ? 'text-center items-center' : 'text-left items-start';

  return (
    <div className={`flex flex-col mb-8 ${alignClass}`}>
      {label && (
        <span className="label-badge mb-2" style={{ color: 'rgba(255,0,0,0.75)' }}>
          {label}
        </span>
      )}
      <h2 className="section-heading text-2xl md:text-3xl">{title}</h2>
      {subtitle && (
        <p className="mt-2 max-w-2xl text-sm md:text-base" style={{ color: '#a1a1aa' }}>
          {subtitle}
        </p>
      )}
      {align === 'center' && <div className="section-underline mt-4" />}
    </div>
  );
}
