{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
  :members:
  :special-members:
  :show-inheritance:


  {% set attr_ns = namespace(inherited=false) %}
  {% for item in attributes %}
    {% if item in inherited_members %}
    {% set attr_ns.inherited = true %}
    {% endif %}
  {%- endfor %}

  {% if attr_ns.inherited %}
  .. rubric:: {{ _('Inherited Attributes') }}
  .. autosummary::
  {% for item in attributes %}
    {% if item in inherited_members %}
    ~{{ name }}.{{ item }}
    {% endif %}
  {%- endfor %}
  {% endif %}

  {% set method_ns = namespace(methods_ext=methods, own=false, inherited=false) %}
  {% for special in ["__call__", "__enter__", "__exit__"] %}
  {% if special in members %}
  {% set method_ns.methods_ext = method_ns.methods_ext + [special] %}
  {% endif %}
  {%- endfor %}
  {% for item in method_ns.methods_ext %}
    {% if item in inherited_members %}
    {% set method_ns.inherited = true %}
    {% else %}
    {% set method_ns.own = true %}
    {% endif %}
  {%- endfor %}

  {% if method_ns.own %}
  .. rubric:: {{ _('Methods') }}
  .. autosummary::
  {% for item in method_ns.methods_ext %}
    {% if item not in inherited_members %}
    ~{{ name }}.{{ item }}
    {% endif %}
  {%- endfor %}
  {% endif %}

  {% if attributes %}
  .. rubric:: {{ _('Attributes') }}
  .. autosummary::
  {% for item in attributes %}
    ~{{ name }}.{{ item }}
  {%- endfor %}
  {% endif %}

  {% if method_ns.inherited %}

  .. rubric:: {{ _('Inherited Methods') }}
  .. raw:: html

    <details style="margin-bottom: 1.5rem">
      <summary style="font-style:italic">(expand to view inherited methods)</summary>

  .. autosummary::
  {% for item in method_ns.methods_ext %}
    {% if item in inherited_members %}
    ~{{ name }}.{{ item }}
    {% endif %}
  {%- endfor %}

  .. raw:: html

    </details>

  {% endif %}
