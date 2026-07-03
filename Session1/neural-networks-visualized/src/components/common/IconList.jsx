/**
 * Bulleted list where every item is prefixed with the same icon — used for
 * "What to Observe" checklists, "Challenge Yourself" prompts, and similar
 * short lists across experiment pages.
 */
function IconList({ items, icon: Icon, iconClassName = "text-emerald-400" }) {
  return (
    <ul className="mt-3 space-y-2 text-sm text-slate-400">
      {items.map((item) => (
        <li key={item} className="flex gap-2">
          <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${iconClassName}`} />
          {item}
        </li>
      ))}
    </ul>
  );
}

export default IconList;
